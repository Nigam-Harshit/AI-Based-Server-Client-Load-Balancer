"""Central HTTP Load Balancer."""

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional
import urllib.error
import urllib.request

from config.backends import DEFAULT_BACKENDS
from load_balancer.router import BaseRouter, get_router
from monitoring.collector import MetricsCollector

logger = logging.getLogger("load_balancer")


class LoadBalancerRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that routes and forwards client requests to backend servers."""

    def log_message(self, format, *args):
        # Suppress default server log to avoid duplicate stdout noise; custom logging is handled in do_GET
        pass

    def _extract_client_ip(self) -> str:
        """Extract the client IP address from request headers or socket address."""
        forwarded_for = self.headers.get("X-Forwarded-For")
        if forwarded_for:
            # First IP in X-Forwarded-For list is the original client
            return forwarded_for.split(",")[0].strip()
        return self.client_address[0]

    def _send_json(self, status_code: int, data: dict, extra_headers: Optional[dict] = None):
        try:
            response_bytes = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(response_bytes)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_GET(self):
        # Direct metrics endpoint for the load balancer to report all backend metrics
        if self.path in ("/lb-metrics", "/metrics/all"):
            collector: MetricsCollector = getattr(self.server, "collector", None)
            if collector:
                self._send_json(200, collector.get_summary())
            else:
                self._send_json(500, {"error": "Metrics collector not initialized"})
            return

        router: BaseRouter = getattr(self.server, "router", None)
        algorithm_name: str = getattr(self.server, "algorithm", "unknown")
        backend_timeout: float = getattr(self.server, "backend_timeout", 2.0)
        client_ip = self._extract_client_ip()
        start_time = time.time()

        if not router:
            self._send_json(500, {"error": "Load balancer router not initialized"})
            return

        with router.route(client_ip=client_ip) as backend:
            target_url = f"{backend.rstrip('/')}{self.path}"
            status_code = 502
            try:
                req = urllib.request.Request(
                    url=target_url,
                    headers={"X-Forwarded-For": client_ip, "User-Agent": "LoadBalancer/1.0"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=backend_timeout) as resp:
                    status_code = resp.status
                    body = resp.read()
                    content_type = resp.headers.get("Content-Type", "application/json")

                    try:
                        self.send_response(status_code)
                        self.send_header("Content-Type", content_type)
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("X-Backend-Server", backend)
                        self.end_headers()
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass

            except urllib.error.HTTPError as e:
                # Backend returned 4xx or 5xx response
                status_code = e.code
                body = e.read()
                content_type = e.headers.get("Content-Type", "application/json")

                try:
                    self.send_response(status_code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Backend-Server", backend)
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            except urllib.error.URLError as e:
                status_code = 502
                self._send_json(
                    502,
                    {
                        "error": "Bad Gateway",
                        "message": f"Backend server unavailable: {e.reason}",
                        "backend": backend,
                    },
                    extra_headers={"X-Backend-Server": backend},
                )

            except TimeoutError:
                status_code = 504
                self._send_json(
                    504,
                    {
                        "error": "Gateway Timeout",
                        "message": "Backend request timed out",
                        "backend": backend,
                    },
                    extra_headers={"X-Backend-Server": backend},
                )

            except Exception as e:
                status_code = 502
                self._send_json(
                    502,
                    {
                        "error": "Bad Gateway",
                        "message": str(e),
                        "backend": backend,
                    },
                    extra_headers={"X-Backend-Server": backend},
                )

            finally:
                duration = time.time() - start_time
                logger.info(
                    "[%s] client=%s backend=%s path=%s status=%d duration=%.4fs",
                    algorithm_name,
                    client_ip,
                    backend,
                    self.path,
                    status_code,
                    duration,
                )


def create_load_balancer(
    host: str = "127.0.0.1",
    port: int = 8000,
    algorithm: str = "round_robin",
    backends: Optional[List[str]] = None,
    backend_timeout: float = 2.0,
) -> ThreadingHTTPServer:
    """Create and configure a ThreadingHTTPServer instance for the load balancer."""
    backends_list = list(backends) if backends else list(DEFAULT_BACKENDS)
    router = get_router(algorithm, backends_list)

    server = ThreadingHTTPServer((host, port), LoadBalancerRequestHandler)
    server.router = router
    server.algorithm = algorithm
    server.backends = backends_list
    server.backend_timeout = backend_timeout
    server.collector = MetricsCollector(backends=backends_list, timeout=backend_timeout)
    return server


def run_load_balancer(
    host: str = "127.0.0.1",
    port: int = 8000,
    algorithm: str = "round_robin",
    backends: Optional[List[str]] = None,
):
    """Run the load balancer HTTP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    httpd = create_load_balancer(host, port, algorithm, backends)
    logger.info(
        "Starting load balancer on %s:%d using %s with backends: %s",
        host,
        port,
        algorithm,
        httpd.backends,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down load balancer")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Central HTTP Load Balancer")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--algorithm",
        default="round_robin",
        choices=["round_robin", "least_connections", "ip_hash"],
        help="Routing algorithm to use (default: round_robin)",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated list of backend URLs (e.g. http://127.0.0.1:8001,http://127.0.0.1:8002)",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
    run_load_balancer(host=args.host, port=args.port, algorithm=args.algorithm, backends=backends)
