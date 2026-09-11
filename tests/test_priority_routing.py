"""Phase 10 Tests: Priority and Deadline Aware Routing Suite.

Validates priority level parsing, deadline slack calculation, urgency classification,
override logic in PriorityDeadlineRouter, and end-to-end load balancer header propagation.
"""

import time
import unittest
from unittest.mock import MagicMock

from config.backends import DEFAULT_BACKENDS
from load_balancer.priority import (
    PriorityLevel,
    DeadlineUrgency,
    PriorityRequestMetadata,
    PriorityDeadlineRouter,
)
from load_balancer.router import RoundRobinRouter
from monitoring.collector import ServerMetrics


class TestPriorityEnumsAndMetadata(unittest.TestCase):
    """Unit tests for PriorityLevel, DeadlineUrgency, and PriorityRequestMetadata."""

    def test_priority_level_ordering(self):
        self.assertLess(PriorityLevel.LOW, PriorityLevel.NORMAL)
        self.assertLess(PriorityLevel.NORMAL, PriorityLevel.HIGH)
        self.assertLess(PriorityLevel.HIGH, PriorityLevel.CRITICAL)
        self.assertEqual(PriorityLevel.from_string("high"), PriorityLevel.HIGH)
        self.assertEqual(PriorityLevel.from_string("UNKNOWN"), PriorityLevel.NORMAL)

    def test_metadata_defaults(self):
        meta = PriorityRequestMetadata(request_id="req-1")
        self.assertEqual(meta.priority, PriorityLevel.NORMAL)
        self.assertIsNone(meta.deadline)
        self.assertEqual(meta.estimated_duration_ms, 30.0)
        self.assertIsNone(meta.calculate_slack_ms(time.time()))
        self.assertEqual(meta.classify_urgency(time.time()), DeadlineUrgency.SAFE)

    def test_metadata_slack_and_urgency(self):
        now = time.time()
        # Deadline in 110ms with 30ms est duration -> slack = 80ms -> APPROACHING_DEADLINE
        meta = PriorityRequestMetadata(
            request_id="req-2",
            priority=PriorityLevel.HIGH,
            deadline=now + 0.11,
            arrival_time=now,
            estimated_duration_ms=30.0,
        )
        slack = meta.calculate_slack_ms(now)
        self.assertIsNotNone(slack)
        self.assertAlmostEqual(slack, 80.0, delta=10.0)
        self.assertEqual(meta.classify_urgency(now), DeadlineUrgency.APPROACHING_DEADLINE)

        # Deadline in 10ms with 30ms est duration -> slack = -20ms -> DEADLINE_RISK
        meta_late = PriorityRequestMetadata(
            request_id="req-3",
            priority=PriorityLevel.CRITICAL,
            deadline=now + 0.01,
            arrival_time=now,
            estimated_duration_ms=30.0,
        )
        self.assertEqual(meta_late.classify_urgency(now), DeadlineUrgency.DEADLINE_RISK)

    def test_metadata_from_headers(self):
        now = time.time()
        headers = {
            "X-Request-Priority": "CRITICAL",
            "X-Request-Deadline": str(now + 1.0),
            "X-Estimated-Duration": "45.0",
            "X-Arrival-Time": str(now),
        }
        meta = PriorityRequestMetadata.from_headers(headers, request_id="hdr-test")
        self.assertEqual(meta.priority, PriorityLevel.CRITICAL)
        self.assertAlmostEqual(meta.deadline, now + 1.0, places=3)
        self.assertEqual(meta.estimated_duration_ms, 45.0)
        self.assertEqual(meta.classify_urgency(now), DeadlineUrgency.SAFE)


class TestPriorityDeadlineRouter(unittest.TestCase):
    """Unit tests for PriorityDeadlineRouter decision engine."""

    def setUp(self):
        self.backends = ["http://127.0.0.1:8001", "http://127.0.0.1:8002", "http://127.0.0.1:8003"]
        self.mock_collector = MagicMock()
        # Default mock metrics: S1 has lowest latency
        default_metrics = {
            "http://127.0.0.1:8001": ServerMetrics("s1", "http://127.0.0.1:8001", 10.0, 30.0, 1, 15.0, 2.0, 0, True),
            "http://127.0.0.1:8002": ServerMetrics("s2", "http://127.0.0.1:8002", 40.0, 50.0, 3, 60.0, 5.0, 2, True),
            "http://127.0.0.1:8003": ServerMetrics("s3", "http://127.0.0.1:8003", 80.0, 80.0, 8, 150.0, 10.0, 5, True),
        }
        self.mock_collector.get_latest_metrics.return_value = default_metrics
        self.mock_collector.collect_all.side_effect = lambda: self.mock_collector.get_latest_metrics.return_value
        self.underlying_router = RoundRobinRouter(self.backends)
        self.router = PriorityDeadlineRouter(
            underlying_router=self.underlying_router,
            backends=self.backends,
            metrics_collector=self.mock_collector,
        )

    def test_safe_request_preserves_underlying_choice(self):
        # NORMAL priority, SAFE urgency should preserve underlying router's choice
        meta = PriorityRequestMetadata(request_id="safe-1", priority=PriorityLevel.NORMAL)
        chosen = self.router.select(client_ip="127.0.0.1", priority_meta=meta)
        self.assertEqual(chosen, self.backends[0])
        self.assertFalse(self.router.last_decision["priority_override"])
        self.assertFalse(self.router.last_decision["deadline_override"])
        self.assertEqual(self.router.last_decision["routing_reason"], "normal_ml_decision")

    def test_urgent_deadline_overrides_to_fastest_backend(self):
        # Advance underlying router so next choice is S3 (which is slow)
        self.underlying_router.select()  # advances to S2
        self.underlying_router.select()  # advances to S3

        # S3 is slow (150ms), but request has deadline risk requiring fastest backend (S1 at 15ms)
        now = time.time()
        urgent_meta = PriorityRequestMetadata(
            request_id="urgent-1",
            priority=PriorityLevel.HIGH,
            deadline=now + 0.02,  # 20ms deadline with 30ms est duration -> DEADLINE_RISK
            arrival_time=now,
            estimated_duration_ms=30.0,
        )
        chosen = self.router.select(client_ip="127.0.0.1", priority_meta=urgent_meta)
        # Should override to S1
        self.assertEqual(chosen, "http://127.0.0.1:8001")
        self.assertTrue(self.router.last_decision["deadline_override"])
        self.assertEqual(self.router.last_decision["routing_reason"], "deadline_slack_override")

    def test_critical_priority_avoids_congested_server(self):
        # Set underlying router to return S3
        self.underlying_router.select()
        self.underlying_router.select()

        # CRITICAL priority request should not accept congested S3 (conn=8, queue=5)
        crit_meta = PriorityRequestMetadata(request_id="crit-1", priority=PriorityLevel.CRITICAL)
        chosen = self.router.select(client_ip="127.0.0.1", priority_meta=crit_meta)
        self.assertEqual(chosen, "http://127.0.0.1:8001")
        self.assertTrue(self.router.last_decision["priority_override"])

    def test_unhealthy_backend_skipped(self):
        # S1 becomes unhealthy
        now = time.time()
        self.mock_collector.get_latest_metrics.return_value = {
            "http://127.0.0.1:8001": ServerMetrics("s1", "http://127.0.0.1:8001", 0.0, 0.0, 0, 0.0, 0.0, 0, timestamp=now, available=False),
            "http://127.0.0.1:8002": ServerMetrics("s2", "http://127.0.0.1:8002", 20.0, 30.0, 1, 20.0, 2.0, 0, timestamp=now, available=True),
            "http://127.0.0.1:8003": ServerMetrics("s3", "http://127.0.0.1:8003", 40.0, 50.0, 2, 40.0, 3.0, 0, timestamp=now, available=True),
        }
        # S1 was chosen by RR, but is offline
        crit_meta = PriorityRequestMetadata(request_id="crit-2", priority=PriorityLevel.CRITICAL)
        chosen = self.router.select(client_ip="127.0.0.1", priority_meta=crit_meta)
        # Must pick healthy S2
        self.assertEqual(chosen, "http://127.0.0.1:8002")

    def test_route_context_manager(self):
        with self.router.route(client_ip="127.0.0.1") as backend:
            self.assertIn(backend, self.backends)


class TestPriorityLiveIntegration(unittest.TestCase):
    """Integration test validating live load balancer routing with priority_ml."""

    @classmethod
    def setUpClass(cls):
        import threading
        import urllib.request
        from server.app import create_server
        from load_balancer.app import create_load_balancer

        cls.lb_port = 8130
        cls.server_ports = [8131, 8132, 8133]
        cls.backends = [f"http://127.0.0.1:{p}" for p in cls.server_ports]
        cls.servers = []
        cls.server_threads = []

        for i, port in enumerate(cls.server_ports, start=1):
            s = create_server(server_id=f"server-{i}", host="127.0.0.1", port=port)
            t = threading.Thread(target=s.serve_forever, daemon=True)
            t.start()
            cls.servers.append(s)
            cls.server_threads.append(t)

        for port in cls.server_ports:
            cls._wait_for_url(f"http://127.0.0.1:{port}/health")

        cls.lb = create_load_balancer(
            host="127.0.0.1",
            port=cls.lb_port,
            algorithm="priority_adaptive",
            backends=cls.backends,
            model_path="models/logistic_regression.joblib",
        )
        cls.lb_thread = threading.Thread(target=cls.lb.serve_forever, daemon=True)
        cls.lb_thread.start()
        cls._wait_for_url(f"http://127.0.0.1:{cls.lb_port}/health")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.lb.shutdown()
            cls.lb.server_close()
        except Exception:
            pass
        for s in cls.servers:
            try:
                s.shutdown()
                s.server_close()
            except Exception:
                pass

    @classmethod
    def _wait_for_url(cls, url: str, timeout: float = 3.0):
        import urllib.request
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"Service at {url} failed to start within {timeout}s")

    def test_live_priority_headers_and_routing(self):
        """Verify headers X-Request-Priority, X-Deadline-Slack, and routing reason in live responses."""
        import urllib.request
        import json

        now = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.01")
        req.add_header("X-Request-Priority", "HIGH")
        req.add_header("X-Request-Deadline", str(now + 0.5))
        req.add_header("X-Estimated-Duration", "10.0")

        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            headers = resp.headers
            self.assertIn("X-Backend-Server", headers)
            self.assertIn("X-Request-Priority", headers)
            self.assertEqual(headers["X-Request-Priority"], "HIGH")
            self.assertIn("X-Deadline-Slack", headers)
            self.assertIn("X-Routing-Reason", headers)
            self.assertIn("X-Final-Backend", headers)
            body = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(body["status"], "completed")


if __name__ == "__main__":
    unittest.main()
