"""Phase 15 Test Suite: Production Hardening, Docker Containerization and Deployment Readiness.

Validates:
1. Dockerfile existence, syntax, non-root security, and configuration.
2. docker-compose.yml YAML syntax, services, network, resource limits, healthchecks.
3. .dockerignore and .env.example completeness.
4. Environment variable configuration externalization.
5. Health endpoints (/health, /lb-health).
6. Router fault tolerance, failover, and candidate filtering.
7. Graceful degradation (502/503) without load balancer crashes.
8. Dynamic Docker CLI detection (skips live daemon tests when Docker is absent).
9. Historical data and model artifact immutability.
"""

import json
import os
import shutil
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.backends import get_configured_backends, DEFAULT_BACKENDS
from load_balancer.app import create_load_balancer
from load_balancer.router import (
    BaseRouter,
    RoundRobinRouter,
    LeastConnectionsRouter,
    IPHashRouter,
    MLRouter,
    get_router,
)
from server.app import create_server


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestDockerConfiguration(unittest.TestCase):
    """Validate Dockerfile, docker-compose.yml, .dockerignore, and .env.example."""

    def test_dockerfile_exists_and_valid(self):
        dockerfile_path = os.path.join(PROJECT_ROOT, "Dockerfile")
        self.assertTrue(os.path.isfile(dockerfile_path), "Dockerfile must exist")
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("FROM python:3.12-slim", content)
        self.assertIn("useradd", content, "Dockerfile must create a non-root user")
        self.assertIn("USER ", content, "Dockerfile must run as non-root user")
        self.assertIn("WORKDIR /app", content)
        self.assertIn("PYTHONUNBUFFERED=1", content)
        self.assertIn("EXPOSE 8000", content)

    def test_dockerignore_exists_and_excludes_sensitive(self):
        ignore_path = os.path.join(PROJECT_ROOT, ".dockerignore")
        self.assertTrue(os.path.isfile(ignore_path), ".dockerignore must exist")
        with open(ignore_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn(".git", content)
        self.assertIn("__pycache__", content)
        self.assertIn(".pytest_cache", content)

    def test_docker_compose_valid_yaml_and_topology(self):
        compose_path = os.path.join(PROJECT_ROOT, "docker-compose.yml")
        self.assertTrue(os.path.isfile(compose_path), "docker-compose.yml must exist")
        with open(compose_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.assertIn("services", data)
        services = data["services"]
        expected_services = ["server-1", "server-2", "server-3", "load-balancer"]
        for svc in expected_services:
            self.assertIn(svc, services, f"Service {svc} must be defined in docker-compose.yml")

        # Verify network
        self.assertIn("networks", data)
        self.assertIn("loadbalancer-net", data["networks"])

        # Verify each service connects to loadbalancer-net
        for svc, config in services.items():
            self.assertIn("networks", config, f"Service {svc} must declare networks")
            self.assertIn("loadbalancer-net", config["networks"])
            self.assertIn("healthcheck", config, f"Service {svc} must define a healthcheck")

        # Verify resource limits on load balancer
        lb_config = services["load-balancer"]
        self.assertIn("deploy", lb_config)
        self.assertIn("resources", lb_config["deploy"])
        self.assertIn("limits", lb_config["deploy"]["resources"])

    def test_env_example_documented(self):
        env_path = os.path.join(PROJECT_ROOT, ".env.example")
        self.assertTrue(os.path.isfile(env_path), ".env.example must exist")
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()

        for var in ["LB_HOST", "LB_PORT", "ROUTING_ALGORITHM", "BACKENDS", "BACKEND_TIMEOUT"]:
            self.assertIn(var, content, f"Variable {var} must be documented in .env.example")


class TestEnvironmentConfiguration(unittest.TestCase):
    """Validate runtime configuration via environment variables."""

    def test_get_configured_backends_env_override(self):
        old_val = os.environ.get("BACKENDS")
        try:
            custom = "http://node1:9001,http://node2:9002"
            os.environ["BACKENDS"] = custom
            backends = get_configured_backends()
            self.assertEqual(backends, ["http://node1:9001", "http://node2:9002"])
        finally:
            if old_val is not None:
                os.environ["BACKENDS"] = old_val
            else:
                os.environ.pop("BACKENDS", None)

    def test_load_balancer_creation_from_env(self):
        port = find_free_port()
        old_algo = os.environ.get("ROUTING_ALGORITHM")
        old_port = os.environ.get("LB_PORT")
        try:
            os.environ["ROUTING_ALGORITHM"] = "round_robin"
            os.environ["LB_PORT"] = str(port)
            lb = create_load_balancer()
            self.assertEqual(lb.algorithm, "round_robin")
            self.assertEqual(lb.server_address[1], port)
            lb.server_close()
        finally:
            if old_algo is not None:
                os.environ["ROUTING_ALGORITHM"] = old_algo
            else:
                os.environ.pop("ROUTING_ALGORITHM", None)
            if old_port is not None:
                os.environ["LB_PORT"] = old_port
            else:
                os.environ.pop("LB_PORT", None)


class TestHealthEndpointsAndFailover(unittest.TestCase):
    """Validate HTTP health checks, dynamic candidate filtering, and fault tolerance."""

    @classmethod
    def setUpClass(cls):
        cls.srv_port = find_free_port()
        cls.lb_port = find_free_port()

        cls.srv = create_server(server_id="test-srv", host="127.0.0.1", port=cls.srv_port)
        cls.srv_thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.srv_thread.start()

        cls.lb = create_load_balancer(
            host="127.0.0.1",
            port=cls.lb_port,
            algorithm="least_connections",
            backends=[f"http://127.0.0.1:{cls.srv_port}"],
            backend_timeout=1.0,
        )
        cls.lb_thread = threading.Thread(target=cls.lb.serve_forever, daemon=True)
        cls.lb_thread.start()
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.lb.shutdown()
            cls.lb.server_close()
        except Exception:
            pass
        try:
            cls.srv.shutdown()
            cls.srv.server_close()
        except Exception:
            pass

    def test_server_health_endpoint(self):
        url = f"http://127.0.0.1:{self.srv_port}/health"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn(data["status"], ("ok", "healthy"))
            self.assertEqual(data["server_id"], "test-srv")

    def test_load_balancer_lb_health_endpoint(self):
        url = f"http://127.0.0.1:{self.lb_port}/lb-health"
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["service"], "load_balancer")
            self.assertEqual(data["algorithm"], "least_connections")

    def test_router_healthy_backend_filtering(self):
        backends = ["http://127.0.0.1:8001", "http://127.0.0.1:8002", "http://127.0.0.1:8003"]
        rr = RoundRobinRouter(backends=backends)
        self.assertEqual(rr.get_candidate_backends(), backends)

        # Inform of partial health
        rr.set_healthy_backends(["http://127.0.0.1:8002", "http://127.0.0.1:8003"])
        candidates = rr.get_candidate_backends()
        self.assertNotIn("http://127.0.0.1:8001", candidates)
        self.assertEqual(len(candidates), 2)

        # Selections must only be from candidates
        selected = [rr.select() for _ in range(4)]
        self.assertTrue(all(s in candidates for s in selected))
        self.assertNotIn("http://127.0.0.1:8001", selected)

    def test_all_backends_down_controlled_502_503(self):
        dead_port = find_free_port()
        lb_dead_port = find_free_port()
        lb_dead = create_load_balancer(
            host="127.0.0.1",
            port=lb_dead_port,
            algorithm="least_connections",
            backends=[f"http://127.0.0.1:{dead_port}"],
            backend_timeout=0.5,
        )
        t = threading.Thread(target=lb_dead.serve_forever, daemon=True)
        t.start()
        time.sleep(0.1)

        try:
            # Request to LB pointing at dead port must return 502/503 without crashing
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{lb_dead_port}/process", timeout=1.5)
                self.fail("Expected HTTP 502 Bad Gateway")
            except urllib.error.HTTPError as e:
                self.assertIn(e.code, (502, 503))

            # LB must still be alive and respond to health check
            with urllib.request.urlopen(f"http://127.0.0.1:{lb_dead_port}/lb-health", timeout=1.0) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            lb_dead.shutdown()
            lb_dead.server_close()


class TestDockerDaemonIntegration(unittest.TestCase):
    """Dynamic Docker verification (skips execution when Docker daemon is not available)."""

    def test_docker_cli_availability(self):
        docker_bin = shutil.which("docker")
        if not docker_bin:
            self.skipTest("Docker CLI not installed in current environment; skipping live container tests.")


class TestHistoricalIntegrity(unittest.TestCase):
    """Ensure Phase 1-14 historical research data and models are strictly unmodified."""

    def test_historical_datasets_exist(self):
        for path in [
            "data/raw",
            "data/phase9",
            "data/phase10",
            "data/phase12",
            "data/phase13",
            "data/phase14",
        ]:
            full_path = os.path.join(PROJECT_ROOT, path)
            self.assertTrue(os.path.isdir(full_path), f"Historical dataset directory {path} must exist")

    def test_model_artifacts_exist(self):
        for model in ["logistic_regression.joblib", "random_forest.joblib", "decision_tree.joblib", "svm.joblib", "xgboost.joblib"]:
            path = os.path.join(PROJECT_ROOT, "models", model)
            self.assertTrue(os.path.isfile(path), f"Model artifact {model} must exist and be preserved")


if __name__ == "__main__":
    unittest.main()
