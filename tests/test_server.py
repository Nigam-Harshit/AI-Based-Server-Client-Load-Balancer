import json
import threading
import time
import urllib.error
import urllib.request
import unittest

from server.app import create_server


class TestServerInstances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.servers = []
        cls.threads = []
        cls.server_configs = [
            {"id": "server-1", "port": 8001},
            {"id": "server-2", "port": 8002},
            {"id": "server-3", "port": 8003},
        ]

        for cfg in cls.server_configs:
            server = create_server(server_id=cfg["id"], host="127.0.0.1", port=cfg["port"])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            cls.servers.append(server)
            cls.threads.append(thread)

        # Wait for all servers to be ready
        for cfg in cls.server_configs:
            cls._wait_for_server(cfg["port"])

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers:
            server.shutdown()
            server.server_close()

    @classmethod
    def _wait_for_server(cls, port: int, timeout: float = 3.0):
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"Server on port {port} failed to start within {timeout}s")

    def test_health_endpoints_all_instances(self):
        for cfg in self.server_configs:
            url = f"http://127.0.0.1:{cfg['port']}/health"
            with urllib.request.urlopen(url, timeout=2.0) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers.get("Content-Type"), "application/json")
                data = json.loads(response.read().decode("utf-8"))
                self.assertEqual(data["status"], "ok")
                self.assertEqual(data["server_id"], cfg["id"])
                self.assertEqual(data["port"], cfg["port"])

    def test_process_endpoint_all_instances(self):
        for cfg in self.server_configs:
            url = f"http://127.0.0.1:{cfg['port']}/process?duration=0.01"
            start_time = time.time()
            with urllib.request.urlopen(url, timeout=2.0) as response:
                elapsed = time.time() - start_time
                self.assertEqual(response.status, 200)
                data = json.loads(response.read().decode("utf-8"))
                self.assertEqual(data["status"], "completed")
                self.assertEqual(data["server_id"], cfg["id"])
                self.assertEqual(data["duration"], 0.01)
                self.assertGreaterEqual(elapsed, 0.01)

    def test_not_found_endpoint(self):
        url = "http://127.0.0.1:8001/nonexistent"
        try:
            urllib.request.urlopen(url, timeout=2.0)
            self.fail("Expected HTTPError 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)
            data = json.loads(e.read().decode("utf-8"))
            self.assertEqual(data["error"], "Not Found")
            self.assertEqual(data["path"], "/nonexistent")


if __name__ == "__main__":
    unittest.main()
