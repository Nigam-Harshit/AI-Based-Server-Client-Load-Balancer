"""Unit and integration tests for ML-driven load balancer routing."""

import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
import numpy as np
import pandas as pd

from load_balancer.app import create_load_balancer
from load_balancer.router import MLRouter, LeastConnectionsRouter, get_router
from monitoring.collector import MetricsCollector, ServerMetrics
from server.app import create_server
from ml.features import extract_features_from_metrics
from ml.trainer import train_and_persist_model


class DummyModel:
    """Mock ML model for testing router prediction behavior and probabilities."""

    def __init__(self, fixed_prediction=0, return_proba=True, fail=False):
        self.fixed_prediction = fixed_prediction
        self.return_proba = return_proba
        self.fail = fail
        self.last_X = None

    def predict(self, X):
        self.last_X = X
        if self.fail:
            raise RuntimeError("Model execution failed")
        return np.array([self.fixed_prediction])

    def predict_proba(self, X):
        if not self.return_proba:
            raise AttributeError("predict_proba not supported")
        probs = [0.1, 0.1, 0.1]
        probs[self.fixed_prediction] = 0.8
        return np.array([probs])


class DummyCollector:
    """Mock MetricsCollector returning controlled backend metrics."""

    def __init__(self, metrics_map=None):
        self.metrics_map = metrics_map or {}

    def collect_all(self):
        return self.metrics_map


class TestMLRouterUnit(unittest.TestCase):
    """Unit tests for MLRouter logic, feature extraction, safety fallback, and pluggability."""

    def setUp(self):
        self.backends = [
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002",
            "http://127.0.0.1:8003",
        ]
        self.metrics_all_healthy = {
            "http://127.0.0.1:8001": ServerMetrics(
                server_id="server-1",
                backend_url="http://127.0.0.1:8001",
                cpu_percent=10.0,
                memory_percent=20.0,
                active_connections=1,
                avg_response_time_ms=5.0,
                network_latency_ms=1.2,
                request_queue_length=0,
                timestamp=time.time(),
                available=True,
            ),
            "http://127.0.0.1:8002": ServerMetrics(
                server_id="server-2",
                backend_url="http://127.0.0.1:8002",
                cpu_percent=50.0,
                memory_percent=60.0,
                active_connections=5,
                avg_response_time_ms=25.0,
                network_latency_ms=2.5,
                request_queue_length=0,
                timestamp=time.time(),
                available=True,
            ),
            "http://127.0.0.1:8003": ServerMetrics(
                server_id="server-3",
                backend_url="http://127.0.0.1:8003",
                cpu_percent=85.0,
                memory_percent=80.0,
                active_connections=12,
                avg_response_time_ms=75.0,
                network_latency_ms=3.1,
                request_queue_length=0,
                timestamp=time.time(),
                available=True,
            ),
        }

    def test_feature_extraction_strictly_15_features(self):
        """Verify extract_features_from_metrics produces exact 15-column pre-routing DataFrame."""
        df, avail = extract_features_from_metrics(self.metrics_all_healthy, self.backends)
        self.assertEqual(df.shape, (1, 15))
        self.assertTrue(avail["server-1"])
        self.assertTrue(avail["server-2"])
        self.assertTrue(avail["server-3"])
        self.assertEqual(df["server_1_cpu"].iloc[0], 10.0)
        self.assertEqual(df["server_2_connections"].iloc[0], 5.0)
        self.assertEqual(df["server_3_response_time"].iloc[0], 75.0)
        # Verify no queue length feature in active 15 features
        self.assertNotIn("server_1_queue_length", df.columns)

    def test_ml_router_select_healthy_prediction(self):
        """Verify normal prediction selects predicted backend and captures metadata."""
        collector = DummyCollector(self.metrics_all_healthy)
        model = DummyModel(fixed_prediction=1)  # predicts server-2
        router = MLRouter(backends=self.backends, model=model, collector=collector)

        chosen = router.select()
        self.assertEqual(chosen, "http://127.0.0.1:8002")
        self.assertFalse(router.last_prediction["is_fallback"])
        self.assertEqual(router.last_prediction["predicted_server"], "http://127.0.0.1:8002")
        self.assertAlmostEqual(router.last_prediction["confidence"], 0.8, places=2)
        self.assertIn("inference_latency_ms", router.last_prediction)

    def test_ml_router_fallback_when_predicted_backend_offline(self):
        """Verify safety fallback to LeastConnections when predicted backend is down."""
        # Mark server-2 as unavailable
        metrics = dict(self.metrics_all_healthy)
        metrics["http://127.0.0.1:8002"] = ServerMetrics(
            server_id="server-2",
            backend_url="http://127.0.0.1:8002",
            cpu_percent=0.0,
            memory_percent=0.0,
            active_connections=0,
            avg_response_time_ms=0.0,
            network_latency_ms=0.0,
            request_queue_length=0,
            timestamp=time.time(),
            available=False,
            error="Connection refused",
        )
        collector = DummyCollector(metrics)
        model = DummyModel(fixed_prediction=1)  # predicts server-2 which is offline
        router = MLRouter(backends=self.backends, model=model, collector=collector)

        chosen = router.select()
        # Should fallback away from server-2
        self.assertTrue(router.last_prediction["is_fallback"])
        self.assertIn("unavailable", router.last_prediction["fallback_reason"])
        self.assertNotEqual(chosen, "http://127.0.0.1:8002")

    def test_ml_router_fallback_on_model_exception(self):
        """Verify fallback when model prediction throws exception."""
        collector = DummyCollector(self.metrics_all_healthy)
        model = DummyModel(fail=True)
        router = MLRouter(backends=self.backends, model=model, collector=collector)

        chosen = router.select()
        self.assertTrue(router.last_prediction["is_fallback"])
        self.assertIn("Inference exception", router.last_prediction["fallback_reason"])
        self.assertIn(chosen, self.backends)

    def test_ml_router_fallback_when_collector_missing(self):
        """Verify fallback when collector is None."""
        model = DummyModel(fixed_prediction=0)
        router = MLRouter(backends=self.backends, model=model, collector=None)

        chosen = router.select()
        self.assertTrue(router.last_prediction["is_fallback"])
        self.assertEqual(router.last_prediction["fallback_reason"], "Model or collector unavailable")
        self.assertEqual(chosen, "http://127.0.0.1:8001")

    def test_ml_router_pluggability_random_forest(self):
        """Verify router seamlessly works with alternative Random Forest pipeline."""
        rf_path = "models/random_forest.joblib"
        if not os.path.exists(rf_path):
            train_and_persist_model(model_type="RandomForest", output_dir="models")

        collector = DummyCollector(self.metrics_all_healthy)
        router = MLRouter(backends=self.backends, model_path=rf_path, collector=collector)
        chosen = router.select()
        self.assertIn(chosen, self.backends)
        self.assertIsNotNone(router.last_prediction["confidence"])

    def test_get_router_factory_ml(self):
        """Verify factory returns MLRouter when algorithm='ml'."""
        collector = DummyCollector(self.metrics_all_healthy)
        router = get_router("ml", self.backends, collector=collector)
        self.assertIsInstance(router, MLRouter)


class TestMLRouterIntegration(unittest.TestCase):
    """Integration test with 3 live backend HTTP servers and ML Load Balancer."""

    @classmethod
    def setUpClass(cls):
        cls.server_ports = [8101, 8102, 8103]
        cls.backends = [f"http://127.0.0.1:{p}" for p in cls.server_ports]
        cls.servers = []
        cls.server_threads = []

        # Start 3 backend nodes
        for i, port in enumerate(cls.server_ports, 1):
            srv = create_server(server_id=f"server-{i}", host="127.0.0.1", port=port)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            cls.servers.append(srv)
            cls.server_threads.append(t)

        # Wait for backends
        for port in cls.server_ports:
            cls._wait_for_url(f"http://127.0.0.1:{port}/health")

        # Start ML Load Balancer on port 8100
        cls.lb_port = 8100
        cls.lb = create_load_balancer(
            host="127.0.0.1",
            port=cls.lb_port,
            algorithm="ml",
            backends=cls.backends,
            model_path="models/logistic_regression.joblib",
        )
        cls.lb_thread = threading.Thread(target=cls.lb.serve_forever, daemon=True)
        cls.lb_thread.start()
        cls._wait_for_url(f"http://127.0.0.1:{cls.lb_port}/health")

    @classmethod
    def tearDownClass(cls):
        cls.lb.shutdown()
        cls.lb.server_close()
        for s in cls.servers:
            s.shutdown()
            s.server_close()

    @classmethod
    def _wait_for_url(cls, url: str, timeout: float = 3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"Service at {url} failed to start within {timeout}s")

    def test_live_ml_forwarding_and_headers(self):
        """Verify live request through ML Load Balancer returns 200 and ML observability headers."""
        url = f"http://127.0.0.1:{self.lb_port}/health"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            headers = resp.headers
            self.assertIn("X-Backend-Server", headers)
            self.assertIn("X-ML-Predicted-Server", headers)
            self.assertIn("X-ML-Fallback", headers)
            self.assertEqual(headers["X-ML-Fallback"], "false")
            self.assertIn("X-ML-Confidence", headers)

            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertIn(data["server_id"], ["server-1", "server-2", "server-3"])

    def test_live_ml_forwarding_process_endpoint(self):
        """Verify live /process request functions through ML Load Balancer."""
        url = f"http://127.0.0.1:{self.lb_port}/process?duration=0.01"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "completed")

    def test_live_ml_fallback_when_backend_fails(self):
        """Verify router safely falls back without throwing when a backend is down."""
        # Temporarily shut down server-1 (port 8101)
        srv1 = self.servers[0]
        srv1.shutdown()
        srv1.server_close()

        try:
            url = f"http://127.0.0.1:{self.lb_port}/health"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                headers = resp.headers
                self.assertIn("X-Backend-Server", headers)
                # Chosen server must be one of the remaining live backends (server-2 or server-3)
                self.assertIn(headers["X-Backend-Server"], ["http://127.0.0.1:8102", "http://127.0.0.1:8103"])
        finally:
            # Restart server-1
            restarted_srv1 = create_server(server_id="server-1", host="127.0.0.1", port=8101)
            t = threading.Thread(target=restarted_srv1.serve_forever, daemon=True)
            t.start()
            self.servers[0] = restarted_srv1
            self.server_threads[0] = t
            self._wait_for_url("http://127.0.0.1:8101/health")


if __name__ == "__main__":
    unittest.main()

