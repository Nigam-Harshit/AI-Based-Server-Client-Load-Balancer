"""Tests for real-time server metric collection."""

import json
import threading
import time
import urllib.request
import unittest

from load_balancer.app import create_load_balancer
from monitoring.collector import MetricsCollector, ServerMetrics
from server.app import create_server


class TestMetricCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        cls.server_threads = []
        cls.server_ports = [8001, 8002, 8003]
        cls.backends = [f"http://127.0.0.1:{p}" for p in cls.server_ports]

        for i, port in enumerate(cls.server_ports, 1):
            server = create_server(server_id=f"server-{i}", host="127.0.0.1", port=port)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            cls.servers.append(server)
            cls.server_threads.append(t)

        for port in cls.server_ports:
            cls._wait_for_service(f"http://127.0.0.1:{port}/health")

        # Load balancer
        cls.lb_port = 8000
        cls.lb = create_load_balancer(
            host="127.0.0.1",
            port=cls.lb_port,
            algorithm="round_robin",
            backends=cls.backends,
        )
        cls.lb_thread = threading.Thread(target=cls.lb.serve_forever, daemon=True)
        cls.lb_thread.start()
        cls._wait_for_service(f"http://127.0.0.1:{cls.lb_port}/health")

    @classmethod
    def tearDownClass(cls):
        cls.lb.shutdown()
        cls.lb.server_close()
        for server in cls.servers:
            server.shutdown()
            server.server_close()

    @classmethod
    def _wait_for_service(cls, url: str, timeout: float = 3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"Service at {url} failed to start within {timeout}s")

    def test_metrics_endpoint_structure_and_types(self):
        """Verify /metrics returns valid data and all 6 metrics have valid types and ranges."""
        url = "http://127.0.0.1:8001/metrics"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))

        required_metrics = [
            "cpu_percent",
            "memory_percent",
            "active_connections",
            "avg_response_time_ms",
            "network_latency_ms",
            "request_queue_length",
        ]
        for metric in required_metrics:
            self.assertIn(metric, data, f"Missing metric {metric} in response")

        # Verify types and ranges
        self.assertIsInstance(data["cpu_percent"], (float, int))
        self.assertTrue(0.0 <= data["cpu_percent"] <= 100.0)

        self.assertIsInstance(data["memory_percent"], (float, int))
        self.assertTrue(0.0 <= data["memory_percent"] <= 100.0)

        self.assertIsInstance(data["active_connections"], int)
        self.assertGreaterEqual(data["active_connections"], 0)

        self.assertIsInstance(data["avg_response_time_ms"], (float, int))
        self.assertGreaterEqual(data["avg_response_time_ms"], 0.0)

        self.assertIsInstance(data["network_latency_ms"], (float, int))
        self.assertGreaterEqual(data["network_latency_ms"], 0.0)

        self.assertIsInstance(data["request_queue_length"], int)
        self.assertGreaterEqual(data["request_queue_length"], 0)

    def test_collector_queries_all_three_servers(self):
        """Verify MetricsCollector successfully queries and aggregates metrics from all 3 servers."""
        collector = MetricsCollector(backends=self.backends)
        metrics = collector.collect_all()

        self.assertEqual(len(metrics), 3)
        for backend_url, metric in metrics.items():
            self.assertIsInstance(metric, ServerMetrics)
            self.assertTrue(metric.available)
            self.assertIn(metric.server_id, ["server-1", "server-2", "server-3"])
            self.assertGreaterEqual(metric.network_latency_ms, 0.0)
            self.assertTrue(0.0 <= metric.memory_percent <= 100.0)

    def test_metrics_change_with_activity(self):
        """Verify average response time and active metrics change when workloads are executed."""
        target_port = 8001
        process_url = f"http://127.0.0.1:{target_port}/process?duration=0.04"

        # Execute a request with known processing duration
        with urllib.request.urlopen(process_url, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)

        # Check metrics on that server
        metrics_url = f"http://127.0.0.1:{target_port}/metrics"
        with urllib.request.urlopen(metrics_url, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # avg_response_time_ms should now reflect the completed ~40ms request
        self.assertGreaterEqual(data["avg_response_time_ms"], 30.0)

    def test_concurrency_and_queue_length_metrics(self):
        """Verify active_connections and request_queue_length respond accurately under load."""
        test_port = 8020
        # Dedicated server with max_workers=1 to reliably trigger queueing
        server = create_server(server_id="server-queue-test", host="127.0.0.1", port=test_port, max_workers=1)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            self._wait_for_service(f"http://127.0.0.1:{test_port}/health")

            def send_slow_request():
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{test_port}/process?duration=0.25", timeout=3.0)
                except Exception:
                    pass

            # Start 2 concurrent requests
            t1 = threading.Thread(target=send_slow_request)
            t2 = threading.Thread(target=send_slow_request)
            t1.start()
            t2.start()

            # Allow threads to reach server
            time.sleep(0.06)

            # Query metrics while both requests are in flight
            with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/metrics", timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # With max_workers=1: 1 active worker, 1 queued waiting
            self.assertEqual(data["active_connections"], 1)
            self.assertEqual(data["request_queue_length"], 1)

            t1.join()
            t2.join()

            # After completion, active and queue return to 0
            with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/metrics", timeout=2.0) as resp:
                data_after = json.loads(resp.read().decode("utf-8"))

            self.assertEqual(data_after["active_connections"], 0)
            self.assertEqual(data_after["request_queue_length"], 0)
        finally:
            server.shutdown()
            server.server_close()

    def test_collector_graceful_handling_unavailable_backend(self):
        """Verify collector handles unreachable/offline servers cleanly without crashing."""
        unavailable = "http://127.0.0.1:8999"
        collector = MetricsCollector(backends=[unavailable], timeout=1.0)
        metric = collector.collect_from_server(unavailable)

        self.assertFalse(metric.available)
        self.assertIsNotNone(metric.error)
        self.assertEqual(metric.cpu_percent, 0.0)
        self.assertEqual(metric.active_connections, 0)
        self.assertEqual(metric.request_queue_length, 0)

    def test_load_balancer_metrics_endpoint(self):
        """Verify load balancer exposes aggregated metrics for all backends via /lb-metrics."""
        url = f"http://127.0.0.1:{self.lb_port}/lb-metrics"
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            summary = json.loads(resp.read().decode("utf-8"))

        self.assertEqual(len(summary), 3)
        for backend_url in self.backends:
            self.assertIn(backend_url, summary)
            server_data = summary[backend_url]
            self.assertTrue(server_data["available"])
            self.assertIn("cpu_percent", server_data)
            self.assertIn("memory_percent", server_data)
            self.assertIn("active_connections", server_data)
            self.assertIn("avg_response_time_ms", server_data)
            self.assertIn("network_latency_ms", server_data)
            self.assertIn("request_queue_length", server_data)


if __name__ == "__main__":
    unittest.main()
