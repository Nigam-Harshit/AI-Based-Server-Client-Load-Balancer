"""Comprehensive Test Suite for Phase 16 Demonstration, Observability & Control Console.

Validates all 32 critical contract, server, routing, telemetry, and fault-injection criteria.
"""

import json
import os
import shutil
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

from client.workload_generator import RequestRecord, WorkloadConfig, WorkloadGenerator
from config.backends import DEFAULT_BACKENDS
from demo.cluster_manager import ClusterManager
from demo.server import create_demo_server
from load_balancer.app import create_load_balancer
from load_balancer.router import (
    BaseRouter,
    IPHashRouter,
    LeastConnectionsRouter,
    MLRouter,
    RoundRobinRouter,
    get_router,
)
from server.app import create_server


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestPhase16DemoSuite(unittest.TestCase):
    """Test suite for Phase 16 components, REST APIs, and end-to-end integration."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="phase16_test_")

        # Start 2 mock backends on dynamic free ports
        cls.backend_port_1 = get_free_port()
        cls.backend_port_2 = get_free_port()
        cls.backend_url_1 = f"http://127.0.0.1:{cls.backend_port_1}"
        cls.backend_url_2 = f"http://127.0.0.1:{cls.backend_port_2}"
        cls.backends = [cls.backend_url_1, cls.backend_url_2]

        cls.srv_b1 = create_server(host="127.0.0.1", port=cls.backend_port_1)
        cls.srv_b2 = create_server(host="127.0.0.1", port=cls.backend_port_2)

        cls.th_b1 = threading.Thread(target=cls.srv_b1.serve_forever, daemon=True)
        cls.th_b2 = threading.Thread(target=cls.srv_b2.serve_forever, daemon=True)
        cls.th_b1.start()
        cls.th_b2.start()

        # Start Load Balancer on dynamic port
        cls.lb_port = get_free_port()
        cls.lb_url = f"http://127.0.0.1:{cls.lb_port}"
        cls.lb_server = create_load_balancer(
            host="127.0.0.1",
            port=cls.lb_port,
            algorithm="round_robin",
            backends=cls.backends,
        )
        cls.th_lb = threading.Thread(target=cls.lb_server.serve_forever, daemon=True)
        cls.th_lb.start()

        # Start Demo Console Server on dynamic port
        cls.demo_port = get_free_port()
        cls.demo_url = f"http://127.0.0.1:{cls.demo_port}"
        cls.demo_server = create_demo_server(
            host="127.0.0.1",
            port=cls.demo_port,
            lb_url=cls.lb_url,
            backends=cls.backends,
        )
        # Override data dir to temp
        cls.demo_server.cluster_mgr.data_dir = cls.temp_dir
        os.makedirs(os.path.join(cls.temp_dir, "live_runs"), exist_ok=True)
        os.makedirs(os.path.join(cls.temp_dir, "experiments"), exist_ok=True)
        os.makedirs(os.path.join(cls.temp_dir, "scenarios"), exist_ok=True)

        # Copy preset scenarios into temp dir
        preset_src = "data/phase16/scenarios/preset_scenarios.json"
        if os.path.exists(preset_src):
            shutil.copy(preset_src, os.path.join(cls.temp_dir, "scenarios", "preset_scenarios.json"))

        cls.demo_server.cluster_mgr._local_servers[cls.backend_port_1] = cls.srv_b1
        cls.demo_server.cluster_mgr._local_servers[cls.backend_port_2] = cls.srv_b2

        cls.th_demo = threading.Thread(target=cls.demo_server.serve_forever, daemon=True)
        cls.th_demo.start()

        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.demo_server.shutdown()
            cls.demo_server.server_close()
        except Exception:
            pass

        try:
            cls.lb_server.shutdown()
            cls.lb_server.server_close()
        except Exception:
            pass

        try:
            cls.srv_b1.shutdown()
            cls.srv_b1.server_close()
        except Exception:
            pass

        try:
            cls.srv_b2.shutdown()
            cls.srv_b2.server_close()
        except Exception:
            pass

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # ----------------------------------------------------------------------
    # 1. Router Factory & Model Instantiation Tests
    # ----------------------------------------------------------------------

    def test_01_router_traditional_instantiation(self):
        """Verify traditional routers instantiate correctly."""
        rr = get_router("round_robin", self.backends)
        self.assertIsInstance(rr, RoundRobinRouter)

        lc = get_router("least_connections", self.backends)
        self.assertIsInstance(lc, LeastConnectionsRouter)

        iph = get_router("ip_hash", self.backends)
        self.assertIsInstance(iph, IPHashRouter)

    def test_02_router_ml_models_by_name(self):
        """Verify standard ML models can be instantiated directly by algorithm name."""
        models = ["logistic_regression", "random_forest", "decision_tree", "svm", "xgboost"]
        for m in models:
            router = get_router(m, self.backends)
            self.assertIsInstance(router, MLRouter)
            self.assertEqual(router.backends, self.backends)

    def test_03_router_adaptive_variants(self):
        """Verify adaptive variants instantiate properly."""
        ad_meta = get_router("adaptive_meta", self.backends)
        self.assertEqual(ad_meta.__class__.__name__, "AdaptiveRouter")
        strategy_meta = getattr(ad_meta.strategy, "value", ad_meta.strategy)
        self.assertEqual(strategy_meta, "meta")

    def test_04_router_priority_variants(self):
        """Verify priority-aware wrappers instantiate."""
        p_ad = get_router("priority_adaptive", self.backends)
        self.assertEqual(p_ad.__class__.__name__, "PriorityDeadlineRouter")

    def test_05_router_invalid_name_raises(self):
        """Verify unsupported algorithm names raise ValueError with helpful hint."""
        with self.assertRaises(ValueError) as ctx:
            get_router("nonexistent_super_algo", self.backends)
        self.assertIn("Supported:", str(ctx.exception))

    # ----------------------------------------------------------------------
    # 2. Dynamic Algorithm Switching on Load Balancer Tests
    # ----------------------------------------------------------------------

    def test_06_lb_get_algorithm_endpoint(self):
        """Verify GET /lb-algorithm on load balancer."""
        req = urllib.request.Request(f"{self.lb_url}/lb-algorithm")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("algorithm", data)
            self.assertIn("backends", data)
            self.assertIn("router_class", data)

    def test_07_lb_post_algorithm_switch(self):
        """Verify POST /lb-algorithm dynamically switches algorithm."""
        payload = {"algorithm": "least_connections"}
        req = urllib.request.Request(
            f"{self.lb_url}/lb-algorithm",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("algorithm"), "least_connections")
            self.assertEqual(data.get("router_class"), "LeastConnectionsRouter")

        # Verify through GET
        req_get = urllib.request.Request(f"{self.lb_url}/lb-algorithm")
        with urllib.request.urlopen(req_get, timeout=2.0) as resp:
            data_get = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data_get.get("algorithm"), "least_connections")

    def test_08_lb_post_algorithm_invalid(self):
        """Verify POST /lb-algorithm with invalid algorithm returns 400."""
        payload = {"algorithm": "invalid_algo"}
        req = urllib.request.Request(
            f"{self.lb_url}/lb-algorithm",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2.0)
        self.assertEqual(ctx.exception.code, 400)

    def test_09_lb_post_algorithm_missing_param(self):
        """Verify POST /lb-algorithm with empty JSON returns 400."""
        req = urllib.request.Request(
            f"{self.lb_url}/lb-algorithm",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2.0)
        self.assertEqual(ctx.exception.code, 400)

    # ----------------------------------------------------------------------
    # 3. WorkloadGenerator Cancellation Tests
    # ----------------------------------------------------------------------

    def test_10_workload_generator_stop_event(self):
        """Verify WorkloadGenerator cleanly terminates when stop_event is set."""
        cfg = WorkloadConfig(target_url=self.lb_url, num_requests=50, concurrency=2, request_rate=5.0)
        stop_evt = threading.Event()
        gen = WorkloadGenerator(cfg, stop_event=stop_evt)

        # Set stop event immediately
        stop_evt.set()
        records = gen.run()
        # Should finish immediately with fewer or cancelled records
        self.assertTrue(len(records) < 50)
        if records:
            self.assertTrue(records[0].error is not None)

    # ----------------------------------------------------------------------
    # 4. Demo Console Server Static Assets Tests
    # ----------------------------------------------------------------------

    def test_11_static_index_html(self):
        """Verify GET / serves index.html with text/html."""
        req = urllib.request.Request(f"{self.demo_url}/")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/html", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("AI-Based Server-Client Load Balancer", content)
            self.assertIn("Phase 16 Demonstration Console", content)

    def test_12_static_styles_css(self):
        """Verify GET /styles.css serves valid CSS."""
        req = urllib.request.Request(f"{self.demo_url}/styles.css")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/css", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("dark-theme", content)

    def test_13_static_app_js(self):
        """Verify GET /app.js serves JavaScript."""
        req = urllib.request.Request(f"{self.demo_url}/app.js")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            c_type = resp.headers.get("Content-Type", "")
            self.assertTrue("javascript" in c_type or "text" in c_type)
            content = resp.read().decode("utf-8")
            self.assertIn("fetchClusterStatus", content)

    def test_14_static_nonexistent_file_404(self):
        """Verify GET on non-existent file returns 404."""
        req = urllib.request.Request(f"{self.demo_url}/nonexistent_asset.png")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2.0)
        self.assertEqual(ctx.exception.code, 404)

    # ----------------------------------------------------------------------
    # 5. Demo Server REST API Tests
    # ----------------------------------------------------------------------

    def test_15_api_status(self):
        """Verify GET /api/status returns cluster and load balancer info."""
        req = urllib.request.Request(f"{self.demo_url}/api/status")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("load_balancer", data)
            self.assertIn("backends", data)
            self.assertIn("cluster_mode", data)
            self.assertIn("workload_running", data)
            self.assertEqual(len(data["backends"]), 2)

    def test_16_api_scenarios(self):
        """Verify GET /api/scenarios returns preset scenarios."""
        req = urllib.request.Request(f"{self.demo_url}/api/scenarios")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            scenarios = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(scenarios, list)
            if scenarios:
                self.assertIn("id", scenarios[0])
                self.assertIn("name", scenarios[0])

    def test_17_api_algorithm_switch(self):
        """Verify POST /api/algorithm switches algorithm via Demo server."""
        payload = {"algorithm": "round_robin"}
        req = urllib.request.Request(
            f"{self.demo_url}/api/algorithm",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("algorithm"), "round_robin")

    def test_18_api_algorithm_invalid(self):
        """Verify POST /api/algorithm with unknown algorithm returns 400."""
        payload = {"algorithm": "totally_invalid_algorithm"}
        req = urllib.request.Request(
            f"{self.demo_url}/api/algorithm",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2.0)
        self.assertEqual(ctx.exception.code, 400)

    def test_19_api_workload_lifecycle(self):
        """Verify POST /api/workload/start and stop lifecycle."""
        payload = {
            "scenario": "stable_normal",
            "num_requests": 6,
            "concurrency": 2,
            "request_duration": 0.01,
        }
        req = urllib.request.Request(
            f"{self.demo_url}/api/workload/start",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "started")

        # Wait briefly for completion
        time.sleep(1.0)

        # Query telemetry
        req_tel = urllib.request.Request(f"{self.demo_url}/api/telemetry")
        with urllib.request.urlopen(req_tel, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            tel = json.loads(resp.read().decode("utf-8"))
            self.assertIn("total_requests", tel)
            self.assertIn("backend_distribution", tel)

    def test_20_api_workload_stop(self):
        """Verify POST /api/workload/stop stops execution."""
        req = urllib.request.Request(
            f"{self.demo_url}/api/workload/stop",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)

    def test_21_api_backend_toggle_stop_and_start(self):
        """Verify POST /api/backend/toggle can disable and re-enable a backend."""
        # Toggle stop
        payload_stop = {"backend": self.backend_url_2, "action": "stop"}
        req_stop = urllib.request.Request(
            f"{self.demo_url}/api/backend/toggle",
            data=json.dumps(payload_stop).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_stop, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "stopped")

        # Verify status reports disabled
        mgr = self.demo_server.cluster_mgr
        st = mgr.get_cluster_status()
        b2_stat = next((b for b in st["backends"] if b["url"] == self.backend_url_2), None)
        self.assertIsNotNone(b2_stat)
        self.assertFalse(b2_stat["healthy"])

        # Toggle start
        payload_start = {"backend": self.backend_url_2, "action": "start"}
        req_start = urllib.request.Request(
            f"{self.demo_url}/api/backend/toggle",
            data=json.dumps(payload_start).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_start, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data.get("status"), "running")

    def test_22_api_session_lifecycle(self):
        """Verify starting and ending an experiment session records output."""
        # Start session
        req_s = urllib.request.Request(
            f"{self.demo_url}/api/session/start",
            data=json.dumps({"name": "Test Session", "description": "Verification"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_s, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            s_data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("session_id", s_data)

        # End session
        req_e = urllib.request.Request(
            f"{self.demo_url}/api/session/end",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_e, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            e_data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("session_file", e_data)
            self.assertTrue(os.path.exists(e_data["session_file"]))

    def test_23_api_export_json_and_csv(self):
        """Verify GET /api/export handles json and csv formats."""
        # JSON export
        req_json = urllib.request.Request(f"{self.demo_url}/api/export?format=json")
        with urllib.request.urlopen(req_json, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("application/json", resp.headers.get("Content-Type", ""))

        # CSV export
        req_csv = urllib.request.Request(f"{self.demo_url}/api/export?format=csv")
        with urllib.request.urlopen(req_csv, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/csv", resp.headers.get("Content-Type", ""))
            csv_text = resp.read().decode("utf-8")
            self.assertIn("request_id", csv_text)

    def test_24_api_demo_status(self):
        """Verify GET /api/demo/status returns guided demo progress."""
        req = urllib.request.Request(f"{self.demo_url}/api/demo/status")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("is_running", data)
            self.assertIn("total_steps", data)

    # ----------------------------------------------------------------------
    # 6. Observability & Telemetry Enrichment Tests
    # ----------------------------------------------------------------------

    def test_25_enrich_record_telemetry(self):
        """Verify ClusterManager._enrich_record extracts custom HTTP headers."""
        rec = RequestRecord(
            request_id=101,
            timestamp=time.time(),
            target=f"{self.lb_url}/process",
            request_type="process",
            request_duration=0.02,
            request_start=1.0,
            request_end=1.05,
            success=True,
            status_code=200,
            response_time=50.0,
            backend_server=self.backend_url_1,
            response_headers={
                "X-Backend-Server": self.backend_url_1,
                "X-Routing-Overhead-Ms": "0.145",
                "X-ML-Predicted-Server": self.backend_url_1,
                "X-ML-Confidence": "0.85",
                "X-ML-Fallback": "false",
                "X-Selected-Model": "random_forest",
                "X-Request-Priority": "HIGH",
                "X-Deadline-Slack": "0.12",
            },
        )
        enriched = self.demo_server.cluster_mgr._enrich_record(rec)
        self.assertEqual(enriched["routing_overhead_ms"], 0.145)
        self.assertEqual(enriched["ml_predicted_server"], self.backend_url_1)
        self.assertEqual(enriched["ml_confidence"], 0.85)
        self.assertFalse(enriched["ml_fallback"])
        self.assertEqual(enriched["selected_model"], "random_forest")
        self.assertEqual(enriched["request_priority"], "HIGH")
        self.assertEqual(enriched["deadline_slack_sec"], 0.12)

    def test_26_cors_headers_present(self):
        """Verify CORS preflight and headers allow cross-origin browser requests."""
        req = urllib.request.Request(f"{self.demo_url}/api/status", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")

    def test_27_live_request_routing_through_lb(self):
        """Verify real HTTP request flows through load balancer to backend."""
        req = urllib.request.Request(f"{self.lb_url}/process?duration=0.01")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("X-Backend-Server", resp.headers)
            self.assertIn(resp.headers["X-Backend-Server"], self.backends)

    def test_28_failover_when_one_backend_down(self):
        """Verify router fails over when one backend server is unavailable."""
        mgr = self.demo_server.cluster_mgr
        # Disable backend 1
        mgr.toggle_backend(self.backend_url_1, action="stop")
        time.sleep(0.3)

        # Make 3 requests — all must succeed by routing to backend 2
        for _ in range(3):
            req = urllib.request.Request(f"{self.lb_url}/process?duration=0.01")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-Backend-Server"), self.backend_url_2)

        # Restore backend 1
        mgr.toggle_backend(self.backend_url_1, action="start")
        time.sleep(0.3)

    def test_29_outcome_c_heuristic_overhead_invariant(self):
        """Verify Outcome C: routing overhead for Round Robin is measured, non-negative, and small."""
        req = urllib.request.Request(f"{self.lb_url}/process?duration=0.01")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            overhead_str = resp.headers.get("X-Routing-Overhead-Ms")
            self.assertIsNotNone(overhead_str)
            overhead = float(overhead_str)
            self.assertGreaterEqual(overhead, 0.0)
            # Under local loopback, RoundRobin overhead is typically < 2ms
            self.assertLess(overhead, 50.0)

    def test_30_live_runs_persisted(self):
        """Verify completed workload runs write artifact to live_runs directory."""
        live_dir = os.path.join(self.demo_server.cluster_mgr.data_dir, "live_runs")
        files = os.listdir(live_dir)
        self.assertGreaterEqual(len(files), 1)

    def test_31_environment_json_exists(self):
        """Verify data/phase16/environment.json is valid and contains metadata."""
        env_file = "data/phase16/environment.json"
        self.assertTrue(os.path.exists(env_file))
        with open(env_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("supported_algorithms", data)
            self.assertIn("console_features", data)

    def test_32_docs_phase16_interface_contract_exists(self):
        """Verify docs/phase16_interface_contract.md is present."""
        contract_doc = "docs/phase16_interface_contract.md"
        self.assertTrue(os.path.exists(contract_doc))
        with open(contract_doc, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertTrue("Phase 16" in content and "Demo Contract" in content)
            self.assertTrue("Outcome C" in content)


if __name__ == "__main__":
    unittest.main()
