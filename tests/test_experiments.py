"""Unit and integration tests for Phase 5 Experimental Data Collection."""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import urllib.request
import unittest

from client.workload_generator import WorkloadConfig, WorkloadGenerator
from experiments.recorder import ExperimentRecord, ExperimentRecorder, RECORD_FIELDNAMES
from experiments.runner import ExperimentRunner
from experiments.validator import DataValidator, REQUIRED_COLUMNS
from load_balancer.app import create_load_balancer
from server.app import create_server


class TestExperimentalDataCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp(prefix="test_exp_data_")
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
        for s in cls.servers:
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.test_dir, ignore_errors=True)

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

    def test_record_creation_and_field_schema(self):
        """Verify ExperimentRecord includes all required pre-routing and outcome fields."""
        recorder = ExperimentRecorder(base_dir=self.test_dir)
        recorder.start_experiment("exp_unit_test", "low_traffic", "round_robin")

        pre_snapshot = {
            "timestamp": time.time(),
            "server_1_cpu": 5.0,
            "server_1_memory": 0.2,
            "server_1_connections": 1,
            "server_1_response_time": 10.0,
            "server_1_network_latency": 1.0,
            "server_1_queue_length": 0,
            "server_2_cpu": 12.0,
            "server_2_memory": 0.25,
            "server_2_connections": 3,
            "server_2_response_time": 25.0,
            "server_2_network_latency": 1.2,
            "server_2_queue_length": 1,
            "server_3_cpu": 4.0,
            "server_3_memory": 0.18,
            "server_3_connections": 0,
            "server_3_response_time": 8.0,
            "server_3_network_latency": 0.9,
            "server_3_queue_length": 0,
        }

        rec = recorder.record_observation(
            pre_snapshot=pre_snapshot,
            request_id=1,
            selected_server="http://127.0.0.1:8001",
            actual_response_time=12.5,
            request_success=True,
            request_start=pre_snapshot["timestamp"] + 0.001,
            request_end=pre_snapshot["timestamp"] + 0.0135,
        )

        self.assertEqual(rec.experiment_id, "exp_unit_test")
        self.assertEqual(rec.request_id, 1)
        self.assertEqual(rec.server_1_cpu, 5.0)
        self.assertEqual(rec.server_2_connections, 3)
        self.assertEqual(rec.server_3_queue_length, 0)
        self.assertTrue(rec.request_success)

        # Validate schema columns
        rec_dict = rec.to_dict()
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, rec_dict)

    def test_label_generation_logic(self):
        """Verify labeling identifies lowest cost server or None when tied."""
        recorder = ExperimentRecorder(base_dir=self.test_dir)

        # Case 1: Server 3 has significantly lower latency and connections -> best is server-3
        pre_snapshot = {
            "timestamp": time.time(),
            "server_1_cpu": 15.0,
            "server_1_memory": 0.2,
            "server_1_connections": 2,
            "server_1_response_time": 40.0,
            "server_1_network_latency": 2.0,
            "server_1_queue_length": 1,
            "server_2_cpu": 20.0,
            "server_2_memory": 0.25,
            "server_2_connections": 4,
            "server_2_response_time": 60.0,
            "server_2_network_latency": 2.5,
            "server_2_queue_length": 2,
            "server_3_cpu": 2.0,
            "server_3_memory": 0.15,
            "server_3_connections": 0,
            "server_3_response_time": 5.0,
            "server_3_network_latency": 1.0,
            "server_3_queue_length": 0,
        }
        rec1 = recorder.record_observation(pre_snapshot=pre_snapshot, request_id=1)
        self.assertEqual(rec1.best_server, "server-3")

        # Case 2: Tied / idle cluster (all servers near zero latency and equal) -> label is None
        pre_idle = {
            "timestamp": time.time(),
            "server_1_cpu": 1.0,
            "server_1_memory": 0.1,
            "server_1_connections": 0,
            "server_1_response_time": 0.0,
            "server_1_network_latency": 0.5,
            "server_1_queue_length": 0,
            "server_2_cpu": 1.0,
            "server_2_memory": 0.1,
            "server_2_connections": 0,
            "server_2_response_time": 0.0,
            "server_2_network_latency": 0.5,
            "server_2_queue_length": 0,
            "server_3_cpu": 1.0,
            "server_3_memory": 0.1,
            "server_3_connections": 0,
            "server_3_response_time": 0.0,
            "server_3_network_latency": 0.5,
            "server_3_queue_length": 0,
        }
        rec2 = recorder.record_observation(pre_snapshot=pre_idle, request_id=2)
        self.assertIsNone(rec2.best_server)

    def test_data_validator_rules(self):
        """Verify DataValidator detects range violations, missing fields, duplicates, and ordering."""
        # Valid record
        valid_rec = {col: 0.0 for col in REQUIRED_COLUMNS}
        valid_rec["experiment_id"] = "exp_val"
        valid_rec["request_id"] = 1
        valid_rec["workload_scenario"] = "low_traffic"
        valid_rec["routing_algorithm"] = "round_robin"
        valid_rec["selected_server"] = "http://127.0.0.1:8001"
        valid_rec["request_type"] = "process"
        valid_rec["request_size"] = 0
        valid_rec["concurrency"] = 1
        valid_rec["request_rate"] = 5.0
        valid_rec["request_success"] = True
        valid_rec["server_1_connections"] = 0
        valid_rec["server_2_connections"] = 0
        valid_rec["server_3_connections"] = 0
        valid_rec["server_1_queue_length"] = 0
        valid_rec["server_2_queue_length"] = 0
        valid_rec["server_3_queue_length"] = 0
        valid_rec["best_server"] = "server-1"
        valid_rec["timestamp"] = 100.0
        valid_rec["request_start"] = 100.01
        valid_rec["request_end"] = 100.05

        errors = DataValidator.validate_record(valid_rec)
        self.assertEqual(len(errors), 0)

        # Range violation (CPU > 100)
        invalid_rec = dict(valid_rec)
        invalid_rec["server_1_cpu"] = 150.0
        errors = DataValidator.validate_record(invalid_rec)
        self.assertTrue(any("cpu" in e.lower() for e in errors))

        # Temporal inconsistency (end < start)
        invalid_time_rec = dict(valid_rec)
        invalid_time_rec["request_end"] = 99.0
        errors = DataValidator.validate_record(invalid_time_rec)
        self.assertTrue(any("temporal" in e.lower() for e in errors))

        # Dataset duplicate detection
        dataset = [valid_rec, dict(valid_rec)]  # identical duplicate
        val_res = DataValidator.validate_dataset(dataset)
        self.assertEqual(val_res["duplicate_records"], 1)

    def test_raw_and_processed_export(self):
        """Verify raw and processed datasets are written cleanly without data loss."""
        recorder = ExperimentRecorder(base_dir=self.test_dir)
        recorder.start_experiment("exp_export_test", "low_traffic", "round_robin")

        pre_snapshot = {
            "timestamp": time.time(),
            "server_1_cpu": 5.0,
            "server_1_memory": 0.2,
            "server_1_connections": 1,
            "server_1_response_time": 10.0,
            "server_1_network_latency": 1.0,
            "server_1_queue_length": 0,
            "server_2_cpu": 10.0,
            "server_2_memory": 0.2,
            "server_2_connections": 2,
            "server_2_response_time": 20.0,
            "server_2_network_latency": 1.1,
            "server_2_queue_length": 0,
            "server_3_cpu": 15.0,
            "server_3_memory": 0.2,
            "server_3_connections": 3,
            "server_3_response_time": 30.0,
            "server_3_network_latency": 1.2,
            "server_3_queue_length": 0,
        }

        for i in range(1, 6):
            recorder.record_observation(
                pre_snapshot=pre_snapshot,
                request_id=i,
                selected_server="http://127.0.0.1:8001",
                actual_response_time=15.0,
                request_success=True,
            )

        recorder.stop_experiment()
        raw_csv = recorder.export_raw("exp_export_test")
        proc_csv = recorder.export_processed("exp_export_test")

        self.assertTrue(raw_csv.exists())
        self.assertTrue(proc_csv.exists())

        # Verify CSV contents
        with open(raw_csv, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 5)
            self.assertEqual(reader[0]["experiment_id"], "exp_export_test")
            self.assertEqual(float(reader[0]["server_1_cpu"]), 5.0)

    def test_end_to_end_experiment_collection(self):
        """Full end-to-end collection test: servers + LB + workload generator + recorder + validator."""
        runner = ExperimentRunner(
            lb_host="127.0.0.1",
            lb_port=self.lb_port,
            server_ports=self.server_ports,
            data_dir=self.test_dir,
        )

        res = runner.run_experiment(
            scenario="low_traffic",
            routing_algorithm="round_robin",
            run_number=1,
            num_requests=10,
            concurrency=2,
            seed=42,
            load_balancer_instance=self.lb,
        )

        self.assertEqual(res["total_records"], 10)
        self.assertEqual(res["valid_records"], 10)
        self.assertEqual(res["validation"]["invalid_records"], 0)
        self.assertEqual(res["validation"]["duplicate_records"], 0)

        # Generate quality report
        report = runner.generate_overall_quality_report()
        self.assertEqual(report["total_observations"], 10)
        self.assertEqual(report["failed_requests"], 0)
        self.assertIn("server_1_cpu", report["feature_ranges"])
        self.assertIn("actual_response_time", report["feature_ranges"])


if __name__ == "__main__":
    unittest.main()
