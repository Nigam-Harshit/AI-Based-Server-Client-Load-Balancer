"""Unit and integration tests for Phase 7 ML Baseline & Model Comparison."""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from ml.baseline import evaluate_traditional_baseline
from ml.models import get_model_registry
from ml.evaluation import (
    ModelEvaluator,
    ACTIVE_PRE_ROUTING_FEATURES,
    ZERO_VARIANCE_FEATURES,
    STRICTLY_EXCLUDED_COLUMNS,
    LABEL_NAMES,
)


class TestMLBaselines(unittest.TestCase):
    """Test suite for ML baseline evaluation, pipelines, and diagnostics."""

    @classmethod
    def setUpClass(cls):
        """Build mock experimental dataset with 36 observations across 3 distinct experiment groups."""
        cls.test_dir = tempfile.mkdtemp()

        rows = []
        for exp_idx, exp_id in enumerate(["exp_grp_1", "exp_grp_2", "exp_grp_3"]):
            for i in range(12):
                req_id = exp_idx * 12 + i + 1
                best_srv = LABEL_NAMES[(i + exp_idx) % 3]
                selected_url = f"http://127.0.0.1:800{(i % 3) + 1}"
                row = {
                    "experiment_id": exp_id,
                    "timestamp": 1000.0 + req_id,
                    "request_id": req_id,
                    "workload_scenario": "burst_traffic" if exp_idx == 0 else ("cpu_heavy" if exp_idx == 1 else "dynamic"),
                    "request_type": "process",
                    "request_size": 100,
                    "concurrency": 2,
                    "request_rate": 10.0,
                    "routing_algorithm": "round_robin",
                    # Server 1
                    "server_1_cpu": 10.0 + i * 0.5,
                    "server_1_memory": 0.20,
                    "server_1_connections": i % 2,
                    "server_1_response_time": 30.0 + i,
                    "server_1_network_latency": 10.0 + (i % 3),
                    "server_1_queue_length": 0,
                    # Server 2
                    "server_2_cpu": 5.0 + i * 0.8,
                    "server_2_memory": 0.20,
                    "server_2_connections": 0,
                    "server_2_response_time": 25.0 + i,
                    "server_2_network_latency": 8.0 + (i % 2),
                    "server_2_queue_length": 0,
                    # Server 3
                    "server_3_cpu": 8.0 + i * 0.3,
                    "server_3_memory": 0.20,
                    "server_3_connections": 1 if i % 3 == 0 else 0,
                    "server_3_response_time": 35.0 + i,
                    "server_3_network_latency": 12.0 + (i % 4),
                    "server_3_queue_length": 0,
                    # Outcomes
                    "selected_server": selected_url,
                    "actual_response_time": 45.0 + i * 2,
                    "request_success": True,
                    "best_server": best_srv,
                    "request_start": 1000.0 + req_id + 0.001,
                    "request_end": 1000.0 + req_id + 0.05,
                }
                rows.append(row)

        cls.mock_df = pd.DataFrame(rows)
        cls.mock_csv_path = os.path.join(cls.test_dir, "processed_mock.csv")
        cls.mock_df.to_csv(cls.mock_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_feature_selection_and_preparation(self):
        """Test that active features are selected, zero-variance dropped, outcomes excluded."""
        evaluator = ModelEvaluator(random_state=42)
        X, y, groups = evaluator.prepare_data(self.mock_df)

        # Active features should be 15
        self.assertEqual(len(X.columns), 15)
        for zf in ZERO_VARIANCE_FEATURES:
            self.assertNotIn(zf, X.columns)
        for oc in STRICTLY_EXCLUDED_COLUMNS:
            self.assertNotIn(oc, X.columns)

        # Target label encoding
        self.assertEqual(len(y), len(self.mock_df))
        self.assertTrue(set(y.unique()).issubset({0, 1, 2}))

        # Groups
        self.assertEqual(groups.nunique(), 3)

    def test_baseline_evaluation(self):
        """Test traditional baseline routing evaluation."""
        base_res = evaluate_traditional_baseline(self.mock_df)
        self.assertIn("accuracy", base_res)
        self.assertIn("macro_f1", base_res)
        self.assertIn("confusion_matrix", base_res)
        self.assertGreaterEqual(base_res["accuracy"], 0.0)
        self.assertLessEqual(base_res["accuracy"], 1.0)
        self.assertEqual(base_res["total_samples"], len(self.mock_df))

    def test_model_registry(self):
        """Test model registry retrieval and pipeline structure."""
        registry = get_model_registry(random_state=42)
        expected_models = ["Random Forest", "Decision Tree", "Logistic Regression", "SVM"]
        for m in expected_models:
            self.assertIn(m, registry)

        # Scaler checks for linear/SVM models
        self.assertIn("scaler", registry["Logistic Regression"].named_steps)
        self.assertIn("scaler", registry["SVM"].named_steps)

        # Tree models should not have scaler step
        self.assertFalse(hasattr(registry["Random Forest"], "named_steps"))
        self.assertFalse(hasattr(registry["Decision Tree"], "named_steps"))

    def test_cross_validation_execution(self):
        """Test GroupKFold cross-validation over groups."""
        evaluator = ModelEvaluator(random_state=42)
        X, y, groups = evaluator.prepare_data(self.mock_df)
        registry = get_model_registry(random_state=42)

        model = registry["Decision Tree"]
        res = evaluator.evaluate_model_cv("Decision Tree", model, X, y, groups, n_splits=3)

        self.assertIn("accuracy_mean", res)
        self.assertIn("accuracy_std", res)
        self.assertIn("macro_f1_mean", res)
        self.assertIn("macro_f1_std", res)
        self.assertIn("confusion_matrix", res)
        self.assertEqual(res["fold_count"], 3)
        self.assertEqual(len(res["per_class_f1"]), 3)

    def test_reproducibility(self):
        """Test that evaluations with identical random seeds yield identical metrics."""
        evaluator1 = ModelEvaluator(random_state=42)
        evaluator2 = ModelEvaluator(random_state=42)

        X, y, groups = evaluator1.prepare_data(self.mock_df)
        model1 = get_model_registry(random_state=42)["Random Forest"]
        model2 = get_model_registry(random_state=42)["Random Forest"]

        res1 = evaluator1.evaluate_model_cv("Random Forest", model1, X, y, groups, n_splits=3)
        res2 = evaluator2.evaluate_model_cv("Random Forest", model2, X, y, groups, n_splits=3)

        self.assertAlmostEqual(res1["accuracy_mean"], res2["accuracy_mean"], places=6)
        self.assertAlmostEqual(res1["macro_f1_mean"], res2["macro_f1_mean"], places=6)

    def test_random_forest_analysis(self):
        """Test MDI and permutation feature importance calculation."""
        evaluator = ModelEvaluator(random_state=42)
        X, y, _ = evaluator.prepare_data(self.mock_df)
        rf = get_model_registry(random_state=42)["Random Forest"]

        rf_analysis = evaluator.analyze_random_forest(rf, X, y)
        self.assertIn("gini_importances", rf_analysis)
        self.assertIn("permutation_importances_mean", rf_analysis)
        self.assertEqual(len(rf_analysis["gini_importances"]), 15)
        self.assertEqual(len(rf_analysis["permutation_importances_mean"]), 15)
        self.assertGreater(len(rf_analysis["top_gini_features"]), 0)

    def test_generate_visualizations(self):
        """Test that all 4 ML evaluation plots are rendered and non-empty."""
        evaluator = ModelEvaluator(random_state=42)
        full_results = evaluator.run_full_comparison(data_dir=self.test_dir)

        out_vis = tempfile.mkdtemp()
        try:
            plots = evaluator.generate_visualizations(full_results, output_dir=out_vis)
            self.assertEqual(len(plots), 4)
            for p in plots:
                self.assertTrue(os.path.exists(p))
                self.assertGreater(os.path.getsize(p), 1000)
        finally:
            shutil.rmtree(out_vis)

    def test_generate_markdown_report(self):
        """Test ML evaluation markdown report generation."""
        evaluator = ModelEvaluator(random_state=42)
        full_results = evaluator.run_full_comparison(data_dir=self.test_dir)

        out_md_dir = tempfile.mkdtemp()
        out_md = os.path.join(out_md_dir, "test_ml_report.md")
        try:
            report_text = evaluator.generate_markdown_report(full_results, output_path=out_md)
            self.assertTrue(os.path.exists(out_md))
            self.assertIn("Phase 7 — Machine Learning Baselines", report_text)
            self.assertIn("Consolidated Model Comparison", report_text)
            self.assertIn("Traditional Baseline", report_text)
            self.assertIn("Random Forest Detailed Analysis", report_text)
            self.assertIn("Conclusion & Gate Decision", report_text)
        finally:
            shutil.rmtree(out_md_dir)


if __name__ == "__main__":
    unittest.main()
