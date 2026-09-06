import argparse
import collections
import json
import logging
import socket
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import psutil

logger = logging.getLogger("server")


class ServerRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for server node with real-time metric collection."""

    def __init__(self, request, client_address, server):
        self.server_id = getattr(server, "server_id", "unknown")
        self.server_port = getattr(server, "server_port", 8000)
        super().__init__(request, client_address, server)

    def log_message(self, format, *args):
        logger.info("[%s] %s - %s", self.server_id, self.address_string(), format % args)

    def _send_json(self, status_code: int, data: dict):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _measure_network_latency(self) -> float:
        """Measure loopback TCP handshake latency in milliseconds."""
        now = time.time()
        last_check = getattr(self.server, "_last_latency_check", 0.0)
        last_latency = getattr(self.server, "_last_network_latency_ms", 0.0)

        # Cache for up to 1.0 second to avoid excessive socket creation
        if now - last_check < 1.0 and last_latency > 0:
            return last_latency

        try:
            t0 = time.perf_counter()
            sock = socket.create_connection(
                (self.server.server_host, self.server.server_port),
                timeout=0.5,
            )
            sock.close()
            t1 = time.perf_counter()
            latency = round((t1 - t0) * 1000.0, 3)
        except Exception:
            latency = 0.0

        self.server._last_network_latency_ms = latency
        self.server._last_latency_check = now
        return latency

    def _get_metrics(self) -> dict:
        """Gather real runtime metrics from the server instance."""
        with self.server._lock:
            active = self.server._active_connections
            queue_len = self.server._request_queue_length
            if self.server._response_times:
                avg_resp = round(sum(self.server._response_times) / len(self.server._response_times), 3)
            else:
                avg_resp = 0.0

        proc: psutil.Process = self.server._process
        cpu = proc.cpu_percent(interval=None)
        mem = round(proc.memory_percent(), 3)
        net_lat = self._measure_network_latency()

        return {
            "server_id": self.server_id,
            "port": self.server_port,
            "cpu_percent": cpu,
            "memory_percent": mem,
            "active_connections": active,
            "avg_response_time_ms": avg_resp,
            "network_latency_ms": net_lat,
            "request_queue_length": queue_len,
        }

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/health":
            self._send_json(200, {
                "status": "ok",
                "server_id": self.server_id,
                "port": self.server_port,
            })
        elif path == "/metrics":
            self._send_json(200, self._get_metrics())
        elif path == "/process":
            # Track request in queue while waiting for execution slot
            with self.server._lock:
                self.server._request_queue_length += 1

            acquired = False
            start_req = time.perf_counter()
            try:
                self.server._worker_semaphore.acquire()
                acquired = True
                with self.server._lock:
                    self.server._request_queue_length = max(0, self.server._request_queue_length - 1)
                    self.server._active_connections += 1

                query_params = parse_qs(parsed_url.query)
                duration = 0.05
                if "duration" in query_params:
                    try:
                        duration = max(0.0, float(query_params["duration"][0]))
                    except (ValueError, IndexError):
                        duration = 0.05

                if duration > 0:
                    time.sleep(duration)

                self._send_json(200, {
                    "status": "completed",
                    "server_id": self.server_id,
                    "duration": duration,
                })
            finally:
                if acquired:
                    elapsed_ms = (time.perf_counter() - start_req) * 1000.0
                    with self.server._lock:
                        self.server._active_connections = max(0, self.server._active_connections - 1)
                        self.server._response_times.append(elapsed_ms)
                    self.server._worker_semaphore.release()
                else:
                    with self.server._lock:
                        self.server._request_queue_length = max(0, self.server._request_queue_length - 1)
        else:
            self._send_json(404, {
                "error": "Not Found",
                "path": path,
            })


def create_server(
    server_id: str,
    host: str = "127.0.0.1",
    port: int = 8001,
    max_workers: int = 20,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance with real-time metrics tracking."""
    server = ThreadingHTTPServer((host, port), ServerRequestHandler)
    server.server_id = server_id
    server.server_host = host
    server.server_port = port
    server.max_workers = max_workers
    server._worker_semaphore = threading.Semaphore(max_workers)
    server._lock = threading.Lock()
    server._active_connections = 0
    server._request_queue_length = 0
    server._response_times = collections.deque(maxlen=50)
    server._process = psutil.Process()
    # Prime CPU percent measurement baseline
    server._process.cpu_percent(interval=None)
    server._last_latency_check = 0.0
    server._last_network_latency_ms = 0.0
    return server


def run_server(server_id: str, host: str = "127.0.0.1", port: int = 8001, max_workers: int = 20):
    """Run the server node indefinitely."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    httpd = create_server(server_id, host, port, max_workers=max_workers)
    logger.info("Starting server %s on %s:%d (max_workers=%d)", server_id, host, port, max_workers)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server %s", server_id)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight HTTP Backend Server Node")
    parser.add_argument("--id", default="server-1", help="Unique identifier for the server node")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    parser.add_argument("--workers", type=int, default=20, help="Maximum concurrent workers (default: 20)")
    args = parser.parse_args()

    run_server(server_id=args.id, host=args.host, port=args.port, max_workers=args.workers)
