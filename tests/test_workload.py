"""Unit and integration tests for the Controlled Workload Generator."""

import json
import threading
import time
import urllib.request
import unittest

from client.workload_generator import (
    RequestRecord,
    SCENARIO_TEMPLATES,
    WorkloadConfig,
    WorkloadGenerator,
    WorkloadSummary,
    get_scenario,
)
from load_balancer.app import create_load_balancer
from monitoring.collector import MetricsCollector
from server.app import create_server


class TestWorkloadGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        cls.server_threads = []
        cls.server_ports = [8001, 8002, 8003]
        cls.backends = [f"http://127.0.0.1:{p}" for p in cls.server_ports]

        # Launch 3 backend servers
        for i, port in enumerate(cls.server_ports, 1):
            server = create_server(server_id=f"server-{i}", host="127.0.0.1", port=port)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            cls.servers.append(server)
            cls.server_threads.append(t)

        for port in cls.server_ports:
            cls._wait_for_service(f"http://127.0.0.1:{port}/health")

        # Launch Load Balancer on port 8000
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

    def test_configuration_validation(self):
        """Verify WorkloadConfig raises ValueError on invalid configurations."""
        # Invalid target URL
        with self.assertRaises(ValueError):
            WorkloadConfig(target_url="invalid_url").validate()

        # Non-positive requests
        with self.assertRaises(ValueError):
            WorkloadConfig(num_requests=0).validate()

        # Non-positive concurrency
        with self.assertRaises(ValueError):
            WorkloadConfig(concurrency=0).validate()

        # Negative request duration
        with self.assertRaises(ValueError):
            WorkloadConfig(request_duration=-0.1).validate()

        # Negative burst interval
        with self.assertRaises(ValueError):
            WorkloadConfig(burst_interval=-1.0).validate()

    def test_scenario_templates_and_overrides(self):
        """Verify all 7 scenarios can be loaded and support overrides."""
        expected_scenarios = [
            "low_traffic",
            "medium_traffic",
            "high_traffic",
            "burst_traffic",
            "cpu_heavy",
            "mixed",
            "dynamic",
        ]
        for name in expected_scenarios:
            self.assertIn(name, SCENARIO_TEMPLATES)
            cfg = get_scenario(name, target_url="http://127.0.0.1:8000", num_requests=10)
            self.assertEqual(cfg.scenario_name, name)
            self.assertEqual(cfg.num_requests, 10)

        with self.assertRaises(ValueError):
            get_scenario("non_existent_scenario")

    def test_request_generation_and_counting(self):
        """Verify workload generator produces exactly the specified number of requests."""
        config = WorkloadConfig(
            target_url=f"http://127.0.0.1:{self.lb_port}",
            num_requests=15,
            concurrency=3,
            endpoint="/health",
            scenario_name="test_count",
        )
        gen = WorkloadGenerator(config)
        records = gen.run()

        self.assertEqual(len(records), 15)
        self.assertEqual([r.request_id for r in records], list(range(1, 16)))
        for r in records:
            self.assertTrue(r.success)
            self.assertEqual(r.status_code, 200)
            self.assertGreater(r.response_time, 0.0)

    def test_response_recording_fields_and_summary(self):
        """Verify all fields in RequestRecord and WorkloadSummary are populated accurately."""
        config = WorkloadConfig(
            target_url=f"http://127.0.0.1:{self.lb_port}",
            num_requests=6,
            concurrency=2,
            endpoint="/process",
            request_duration=0.01,
            scenario_name="test_fields",
        )
        gen = WorkloadGenerator(config)
        records = gen.run()
        summary = gen.get_summary()

        self.assertEqual(len(records), 6)
        r0 = records[0]
        self.assertIsInstance(r0, RequestRecord)
        self.assertEqual(r0.request_type, "process")
        self.assertEqual(r0.request_duration, 0.01)
        self.assertTrue(r0.success)
        self.assertEqual(r0.status_code, 200)
        self.assertIsNotNone(r0.backend_server)

        # Verify summary stats
        self.assertIsInstance(summary, WorkloadSummary)
        self.assertEqual(summary.total_requests, 6)
        self.assertEqual(summary.successful_requests, 6)
        self.assertEqual(summary.failed_requests, 0)
        self.assertGreater(summary.actual_throughput_rps, 0.0)
        self.assertGreaterEqual(summary.avg_response_time_ms, summary.min_response_time_ms)
        self.assertLessEqual(summary.avg_response_time_ms, summary.max_response_time_ms)

    def test_burst_traffic_workload(self):
        """Verify burst dispatching generates correct request counts in batches."""
        config = WorkloadConfig(
            target_url=f"http://127.0.0.1:{self.lb_port}",
            num_requests=20,
            concurrency=10,
            burst_size=10,
            burst_interval=0.1,
            endpoint="/process",
            request_duration=0.01,
            scenario_name="test_burst",
        )
        gen = WorkloadGenerator(config)
        records = gen.run()
        self.assertEqual(len(records), 20)
        self.assertTrue(all(r.success for r in records))

    def test_workload_affects_phase3_metrics(self):
        """Verify controlled workload produces measurable changes in Phase-3 metrics."""
        collector = MetricsCollector(backends=self.backends)
        target_server = self.backends[0]

        # Baseline metrics before workload
        baseline_server1 = collector.collect_from_server(target_server)
        self.assertTrue(baseline_server1.available)

        # Run CPU-heavy workload against Server 1 specifically
        heavy_config = WorkloadConfig(
            target_url=target_server,
            num_requests=10,
            concurrency=5,
            endpoint="/process",
            request_duration=0.08,  # 80ms processing per request
            scenario_name="test_heavy",
        )
        gen = WorkloadGenerator(heavy_config)
        records = gen.run()
        self.assertEqual(len(records), 10)

        # Query metrics after workload
        post_metrics = collector.collect_from_server(target_server)
        self.assertTrue(post_metrics.available)
        # Server 1's average response time should increase significantly
        self.assertGreater(post_metrics.avg_response_time_ms, baseline_server1.avg_response_time_ms)
        self.assertGreaterEqual(post_metrics.avg_response_time_ms, 30.0)

    def test_mixed_workload_reproducibility_with_seed(self):
        """Verify random seed provides deterministic request sequences for mixed workloads."""
        cfg1 = WorkloadConfig(
            target_url=f"http://127.0.0.1:{self.lb_port}",
            num_requests=10,
            concurrency=2,
            endpoint="mixed",
            seed=12345,
        )
        gen1 = WorkloadGenerator(cfg1)
        items1 = [gen1._build_request_item(i) for i in range(1, 11)]

        cfg2 = WorkloadConfig(
            target_url=f"http://127.0.0.1:{self.lb_port}",
            num_requests=10,
            concurrency=2,
            endpoint="mixed",
            seed=12345,
        )
        gen2 = WorkloadGenerator(cfg2)
        items2 = [gen2._build_request_item(i) for i in range(1, 11)]

        self.assertEqual(items1, items2)


if __name__ == "__main__":
    unittest.main()
