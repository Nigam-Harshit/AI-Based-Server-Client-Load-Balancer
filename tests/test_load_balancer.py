"""Integration tests for the Load Balancer HTTP server."""

import json
import threading
import time
import urllib.error
import urllib.request
import unittest

from load_balancer.app import create_load_balancer
from server.app import create_server


class TestLoadBalancerIntegration(unittest.TestCase):
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

        # Wait for all 3 backend servers to be ready
        for port in cls.server_ports:
            cls._wait_for_service(f"http://127.0.0.1:{port}/health")

        # Launch Round-Robin Load Balancer on port 8000
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
        # Shut down load balancer
        cls.lb.shutdown()
        cls.lb.server_close()

        # Shut down backend servers
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

    def test_round_robin_forwarding_and_backend_identity(self):
        """Verify sequential requests to Load Balancer are forwarded in round-robin order."""
        # Reset router index to ensure clean start
        self.lb.router._index = 0
        expected_sequence = [
            "server-1",
            "server-2",
            "server-3",
            "server-1",
            "server-2",
            "server-3",
        ]
        actual_sequence = []
        for _ in range(len(expected_sequence)):
            url = f"http://127.0.0.1:{self.lb_port}/health"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("Content-Type"), "application/json")
                self.assertIn("X-Backend-Server", resp.headers)
                data = json.loads(resp.read().decode("utf-8"))
                actual_sequence.append(data["server_id"])
                self.assertEqual(data["status"], "ok")
        self.assertEqual(actual_sequence, expected_sequence)

    def test_forwarding_process_endpoint_with_workload(self):
        """Verify /process endpoint query params and workload are forwarded properly."""
        url = f"http://127.0.0.1:{self.lb_port}/process?duration=0.02"
        start = time.time()
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            elapsed = time.time() - start
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["duration"], 0.02)
            self.assertGreaterEqual(elapsed, 0.02)

    def test_algorithm_switching_least_connections(self):
        """Verify load balancer functions when initialized with least_connections."""
        port = 8010
        lb_lc = create_load_balancer(
            host="127.0.0.1",
            port=port,
            algorithm="least_connections",
            backends=self.backends,
        )
        t = threading.Thread(target=lb_lc.serve_forever, daemon=True)
        t.start()
        try:
            self._wait_for_service(f"http://127.0.0.1:{port}/health")
            url = f"http://127.0.0.1:{port}/health"
            with urllib.request.urlopen(url, timeout=3.0) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["status"], "ok")
                self.assertIn("server_id", data)
        finally:
            lb_lc.shutdown()
            lb_lc.server_close()

    def test_algorithm_switching_ip_hash(self):
        """Verify load balancer with ip_hash consistently routes the same client IP."""
        port = 8011
        lb_iph = create_load_balancer(
            host="127.0.0.1",
            port=port,
            algorithm="ip_hash",
            backends=self.backends,
        )
        t = threading.Thread(target=lb_iph.serve_forever, daemon=True)
        t.start()
        try:
            self._wait_for_service(f"http://127.0.0.1:{port}/health")
            url = f"http://127.0.0.1:{port}/health"

            # Use custom X-Forwarded-For header to simulate specific client IP
            req = urllib.request.Request(url, headers={"X-Forwarded-For": "192.168.1.100"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                first_server = json.loads(resp.read().decode("utf-8"))["server_id"]

            for _ in range(4):
                req = urllib.request.Request(url, headers={"X-Forwarded-For": "192.168.1.100"})
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    server_id = json.loads(resp.read().decode("utf-8"))["server_id"]
                    self.assertEqual(server_id, first_server)
        finally:
            lb_iph.shutdown()
            lb_iph.server_close()

    def test_backend_failure_handling(self):
        """Verify load balancer does not crash when a backend is unreachable and returns 502."""
        port = 8012
        # Configure load balancer pointing only to an unavailable port
        unavailable_backend = "http://127.0.0.1:8999"
        lb_fail = create_load_balancer(
            host="127.0.0.1",
            port=port,
            algorithm="round_robin",
            backends=[unavailable_backend],
            backend_timeout=1.0,
        )
        t = threading.Thread(target=lb_fail.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}/health"
            # Send request expecting 502 Bad Gateway
            try:
                urllib.request.urlopen(url, timeout=5.0)
                self.fail("Expected 502 Bad Gateway")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 502)
                data = json.loads(e.read().decode("utf-8"))
                self.assertEqual(data["error"], "Bad Gateway")
                self.assertIn("unavailable", data["message"].lower())
                self.assertEqual(data["backend"], unavailable_backend)
        finally:
            lb_fail.shutdown()
            lb_fail.server_close()


if __name__ == "__main__":
    unittest.main()
