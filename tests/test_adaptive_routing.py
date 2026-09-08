"""Unit and integration tests for Phase 13: Adaptive / Context-Aware ML Routing.

Validates:
1. Selector initialization, strategy parsing, and model artifact loading.
2. Evidence-Based Policy Selector regime inference and model assignment.
3. Learned Meta-Selector training and prediction lifecycle.
4. Safe fallback to LeastConnectionsRouter upon model outage, low confidence, or exceptions.
5. Zero post-routing feature leakage contract.
6. Model-switching tracking and switch count correctness.
7. End-to-end HTTP load balancer integration with '--algorithm adaptive' and observability headers.
8. Clean isolation of priority and deadline policy layering.
9. Historical dataset immutability (Phases 5, 6, 9, 10, 12 remain unmodified).
10. Phase 13 generated datasets and all 12 publication plots exist and are non-empty.
"""

import json
import os
import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from config.backends import DEFAULT_BACKENDS
from load_balancer.app import create_load_balancer
from load_balancer.router import get_router, LeastConnectionsRouter
from ml.adaptive_selector import (
    AdaptiveRouter,
    AdaptiveStrategy,
    EvidenceBasedPolicySelector,
    LearnedMetaSelector,
    CANDIDATE_MODELS,
    DEFAULT_MODEL_PATHS,
)
from ml.evaluation import ACTIVE_PRE_ROUTING_FEATURES
from monitoring.collector import ServerMetrics


class TestAdaptiveRouting(unittest.TestCase):
    """Test suite for Phase 13 Adaptive ML Routing."""

    def setUp(self):
        self.backends = list(DEFAULT_BACKENDS)
        self.mock_collector = MagicMock()
        import time
        now = time.time()
        self.mock_collector.collect_all.return_value = {
            "http://127.0.0.1:8001": ServerMetrics(
                server_id="server-1",
                backend_url="http://127.0.0.1:8001",
                cpu_percent=20.0,
                memory_percent=35.0,
                active_connections=1,
                avg_response_time_ms=15.0,
                network_latency_ms=0.5,
                request_queue_length=0,
                timestamp=now,
                available=True,
            ),
            "http://127.0.0.1:8002": ServerMetrics(
                server_id="server-2",
                backend_url="http://127.0.0.1:8002",
                cpu_percent=45.0,
                memory_percent=40.0,
                active_connections=3,
                avg_response_time_ms=30.0,
                network_latency_ms=0.6,
                request_queue_length=0,
                timestamp=now,
                available=True,
            ),
            "http://127.0.0.1:8003": ServerMetrics(
                server_id="server-3",
                backend_url="http://127.0.0.1:8003",
                cpu_percent=18.0,
                memory_percent=30.0,
                active_connections=1,
                avg_response_time_ms=14.0,
                network_latency_ms=0.4,
                request_queue_length=0,
                timestamp=now,
                available=True,
            ),
        }

    def test_selector_initialization_and_models(self):
        """Verify adaptive router initializes with candidate models and valid strategy."""
        router = AdaptiveRouter(
            backends=self.backends,
            collector=self.mock_collector,
            strategy="policy",
        )
        self.assertEqual(router.strategy, AdaptiveStrategy.POLICY)
        self.assertIsInstance(router.fallback_router, LeastConnectionsRouter)
        self.assertGreaterEqual(len(router.models), 3, "Should load available candidate models")

    def test_strategy_parsing(self):
        """Verify string strategy parsing."""
        self.assertEqual(AdaptiveStrategy.from_string("policy"), AdaptiveStrategy.POLICY)
        self.assertEqual(AdaptiveStrategy.from_string("meta"), AdaptiveStrategy.META)
        self.assertEqual(AdaptiveStrategy.from_string("learned_meta"), AdaptiveStrategy.META)
        self.assertEqual(AdaptiveStrategy.from_string("UNKNOWN"), AdaptiveStrategy.POLICY)

    def test_policy_selector_regime_inference(self):
        """Verify policy selector correctly infers regimes from pre-routing features."""
        policy = EvidenceBasedPolicySelector()

        # Low load
        df_low = pd.DataFrame([{
            "server_1_cpu": 15.0, "server_2_cpu": 16.0, "server_3_cpu": 14.0,
            "server_1_connections": 0, "server_2_connections": 1, "server_3_connections": 0,
            "server_1_response_time": 12.0, "server_2_response_time": 14.0, "server_3_response_time": 11.0,
        }])
        self.assertEqual(policy.infer_regime(df_low), "low_load")
        sel_m, _, _ = policy.select_model(df_low)
        self.assertEqual(sel_m, "SVM")

        # High load / Queue contention
        df_queue = pd.DataFrame([{
            "server_1_cpu": 60.0, "server_2_cpu": 65.0, "server_3_cpu": 55.0,
            "server_1_connections": 4, "server_2_connections": 4, "server_3_connections": 3,
            "server_1_response_time": 55.0, "server_2_response_time": 60.0, "server_3_response_time": 50.0,
        }])
        self.assertEqual(policy.infer_regime(df_queue), "queue_contention")
        sel_m, _, _ = policy.select_model(df_queue)
        self.assertEqual(sel_m, "Random Forest")

        # Backend Imbalance (high spread)
        df_imbalance = pd.DataFrame([{
            "server_1_cpu": 20.0, "server_2_cpu": 90.0, "server_3_cpu": 15.0,
            "server_1_connections": 2, "server_2_connections": 5, "server_3_connections": 1,
            "server_1_response_time": 18.0, "server_2_response_time": 85.0, "server_3_response_time": 16.0,
        }])
        self.assertEqual(policy.infer_regime(df_imbalance), "backend_imbalance")

    def test_meta_selector_fit_and_predict(self):
        """Verify learned meta-selector can fit and predict best model from context features."""
        meta = LearnedMetaSelector()
        self.assertFalse(meta.is_fitted)

        # Unfitted returns robust default
        m, conf, _ = meta.select_model(pd.DataFrame([{"server_1_cpu": 10.0}]))
        self.assertEqual(m, "SVM")

        # Synthetic fit
        X_mock = pd.DataFrame({feat: np.random.rand(20) * 50 for feat in ACTIVE_PRE_ROUTING_FEATURES})
        y_mock = pd.Series(["SVM"] * 10 + ["Random Forest"] * 10)
        meta.fit(X_mock, y_mock)
        self.assertTrue(meta.is_fitted)

        pred_m, pred_conf, _ = meta.select_model(X_mock.iloc[[0]])
        self.assertIn(pred_m, ["SVM", "Random Forest"])
        self.assertGreaterEqual(pred_conf, 0.0)

    def test_adaptive_routing_decision(self):
        """Verify adaptive router select returns a valid backend URL and populates metadata."""
        router = AdaptiveRouter(
            backends=self.backends,
            collector=self.mock_collector,
            strategy="policy",
        )
        chosen = router.select(client_ip="127.0.0.1")
        self.assertIn(chosen, self.backends)

        dec = router.last_adaptive_decision
        self.assertIn("selected_model", dec)
        self.assertIn("strategy", dec)
        self.assertEqual(dec["strategy"], "policy")
        self.assertIn("model_selection_time_ms", dec)
        self.assertIn("switch_count", dec)
        self.assertIn("is_fallback", dec)

    def test_fallback_when_collector_unavailable(self):
        """Verify graceful fallback to LeastConnectionsRouter when collector is None."""
        router = AdaptiveRouter(backends=self.backends, collector=None)
        chosen = router.select()
        self.assertIn(chosen, self.backends)
        self.assertTrue(router.last_adaptive_decision["is_fallback"])
        self.assertIn("Collector unavailable", router.last_adaptive_decision["fallback_reason"])
        self.assertEqual(router.fallback_count, 1)

    def test_fallback_when_no_backends_healthy(self):
        """Verify fallback when all backends are reported unhealthy."""
        import time
        now = time.time()
        unhealthy_collector = MagicMock()
        unhealthy_collector.collect_all.return_value = {
            b: ServerMetrics(
                server_id=f"server-{i+1}",
                backend_url=b,
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                avg_response_time_ms=0.0,
                network_latency_ms=0.0,
                request_queue_length=0,
                timestamp=now,
                available=False,
                error="Down",
            )
            for i, b in enumerate(self.backends)
        }
        router = AdaptiveRouter(backends=self.backends, collector=unhealthy_collector)
        chosen = router.select()
        self.assertIn(chosen, self.backends)
        self.assertTrue(router.last_adaptive_decision["is_fallback"])
        self.assertEqual(router.last_adaptive_decision["fallback_reason"], "No backends healthy")

    def test_model_switching_tracking(self):
        """Verify router increments switch_count when selected model changes."""
        router = AdaptiveRouter(backends=self.backends, collector=self.mock_collector)
        router.current_model = "SVM"

        # Force next selection to Random Forest
        router.policy_selector.select_model = MagicMock(return_value=("Random Forest", 0.95, "test"))
        router.select()
        self.assertEqual(router.switch_count, 1)
        self.assertEqual(router.current_model, "Random Forest")

        # Same model -> no increment
        router.select()
        self.assertEqual(router.switch_count, 1)

    def test_zero_leakage_feature_contract(self):
        """Verify adaptive selector features are strictly pre-routing metrics."""
        forbidden_leakage_terms = ["response_time_ms", "status_code", "actual_response_time", "request_success", "best_server"]
        for feat in ACTIVE_PRE_ROUTING_FEATURES:
            self.assertNotIn(feat, forbidden_leakage_terms, f"Feature {feat} causes data leakage!")

    def test_router_factory_integration(self):
        """Verify get_router factory instantiates adaptive and priority_adaptive routers."""
        r_adpt = get_router("adaptive", self.backends, collector=self.mock_collector)
        self.assertIsInstance(r_adpt, AdaptiveRouter)

        r_p_adpt = get_router("priority_adaptive", self.backends, collector=self.mock_collector)
        from load_balancer.priority import PriorityDeadlineRouter
        self.assertIsInstance(r_p_adpt, PriorityDeadlineRouter)

    def test_live_http_load_balancer_creation(self):
        """Verify create_load_balancer supports algorithm='adaptive'."""
        server = create_load_balancer(
            host="127.0.0.1",
            port=8099,
            algorithm="adaptive",
            backends=self.backends,
            adaptive_strategy="policy",
        )
        self.assertEqual(server.algorithm, "adaptive")
        self.assertIsInstance(server.router, AdaptiveRouter)
        server.server_close()

    def test_historical_datasets_immutability(self):
        """Verify historical datasets from Phases 5, 6, 9, 10, 12 remain unmodified."""
        historical_checks = {
            "data/raw/experiment_data.csv": 120,
            "data/phase9/raw_results.csv": 4800,
            "data/phase12/phase12_raw.csv": 1710,
        }
        for path, expected_rows in historical_checks.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.assertEqual(len(df), expected_rows, f"Historical file {path} was modified!")

    def test_phase13_artifacts_and_plots_exist(self):
        """Verify all Phase 13 datasets, summary CSVs, metadata, and 12 plots exist."""
        required_files = [
            "data/phase13/meta_dataset.csv",
            "data/phase13/phase13_summary.csv",
            "data/phase13/phase13_raw.csv",
            "data/phase13/experiment_manifest.json",
            "data/phase13/environment.json",
            "data/phase13/statistical_analysis.json",
        ]
        for f in required_files:
            self.assertTrue(os.path.exists(f), f"Required artifact {f} missing")

        plot_files = [
            "01_fixed_vs_adaptive_performance.png",
            "02_model_selection_frequency.png",
            "03_model_selection_by_scenario.png",
            "04_adaptive_vs_fixed_macro_f1.png",
            "05_adaptive_vs_fixed_accuracy.png",
            "06_latency_comparison.png",
            "07_throughput_comparison.png",
            "08_model_switching_timeline.png",
            "09_selector_confusion_matrix.png",
            "10_adaptive_overhead.png",
            "11_fallback_analysis.png",
            "12_priority_adaptive_comparison.png",
        ]
        for p in plot_files:
            p_path = os.path.join("experiments/results/phase13", p)
            self.assertTrue(os.path.exists(p_path), f"Plot {p} missing")
            self.assertGreater(os.path.getsize(p_path), 5000, f"Plot {p} is empty or too small")


if __name__ == "__main__":
    unittest.main()
