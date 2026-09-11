"""Central HTTP Load Balancer."""

import argparse
import json
import logging
import os
import threading
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

        # Direct health endpoint for the load balancer
        if self.path in ("/lb-health", "/healthz", "/ping"):
            self._send_json(200, {
                "status": "ok",
                "service": "load_balancer",
                "algorithm": getattr(self.server, "algorithm", "unknown"),
                "backends": getattr(self.server, "backends", []),
            })
            return

        # Direct algorithm inspection endpoint
        if self.path == "/lb-algorithm":
            router = getattr(self.server, "router", None)
            self._send_json(200, {
                "algorithm": getattr(self.server, "algorithm", "unknown"),
                "backends": getattr(self.server, "backends", []),
                "router_class": router.__class__.__name__ if router else None,
            })
            return

        router: BaseRouter = getattr(self.server, "router", None)
        algorithm_name: str = getattr(self.server, "algorithm", "unknown")
        backend_timeout: float = getattr(self.server, "backend_timeout", 2.0)
        recorder = getattr(self.server, "recorder", None)
        client_ip = self._extract_client_ip()
        start_time = time.time()

        if not router:
            self._send_json(500, {"error": "Load balancer router not initialized"})
            return

        # Capture pre-routing metrics snapshot strictly BEFORE routing decision
        pre_snapshot = None
        if recorder and getattr(recorder, "is_recording", False):
            collector = getattr(self.server, "collector", None)
            if collector:
                pre_snapshot = recorder.capture_pre_snapshot(collector, timestamp=start_time)

        # Extract priority and deadline metadata if present
        from load_balancer.priority import PriorityRequestMetadata
        req_id_hdr = self.headers.get("X-Request-ID") or f"req_{int(start_time*1000)}"
        priority_meta = PriorityRequestMetadata.from_headers(self.headers, request_id=req_id_hdr)

        # Multi-attempt failover loop (allows failing over if a backend server drops connection)
        max_attempts = min(3, max(1, len(getattr(self.server, "backends", [1]))))
        attempt = 0
        handled = False

        while attempt < max_attempts and not handled:
            attempt += 1

            # Refresh healthy backends based on current status
            if hasattr(self.server, "_unhealthy_lock"):
                with self.server._unhealthy_lock:
                    now = time.time()
                    self.server._unhealthy_backends = {
                        b: exp for b, exp in self.server._unhealthy_backends.items() if exp > now
                    }
                    active_healthy = [b for b in self.server.backends if b not in self.server._unhealthy_backends]
                    if hasattr(router, "set_healthy_backends"):
                        router.set_healthy_backends(active_healthy if active_healthy else None)

            # Route request through router (supporting priority_meta if router accepts it)
            from load_balancer.priority import PriorityDeadlineRouter
            t_route_start = time.perf_counter()
            try:
                if isinstance(router, PriorityDeadlineRouter):
                    route_ctx = router.route(client_ip=client_ip, priority_meta=priority_meta)
                else:
                    try:
                        route_ctx = router.route(client_ip=client_ip, priority_meta=priority_meta)
                    except TypeError:
                        route_ctx = router.route(client_ip=client_ip)
            except Exception as e:
                logger.error("Failed to acquire backend route: %s", e)
                self._send_json(503, {
                    "error": "Service Unavailable",
                    "message": "No healthy backend available to route request",
                    "detail": str(e),
                    "status": 503,
                })
                return

            with route_ctx as backend:
                routing_overhead_ms = max(0.0, round((time.perf_counter() - t_route_start) * 1000.0, 3))
                target_url = f"{backend.rstrip('/')}{self.path}"
                status_code = 502

                # Extract ML and Priority observability headers
                extra_response_headers = {
                    "X-Backend-Server": backend,
                    "X-Routing-Overhead-Ms": f"{routing_overhead_ms:.3f}",
                }
                if attempt > 1:
                    extra_response_headers["X-Failover-Attempt"] = str(attempt)

                if hasattr(router, "last_prediction") and router.last_prediction:
                    pred_meta = router.last_prediction
                    if pred_meta.get("predicted_server"):
                        extra_response_headers["X-ML-Predicted-Server"] = str(pred_meta["predicted_server"])
                    if pred_meta.get("confidence") is not None:
                        extra_response_headers["X-ML-Confidence"] = f"{pred_meta['confidence']:.4f}"
                    extra_response_headers["X-ML-Fallback"] = "true" if pred_meta.get("is_fallback") else "false"

                # Extract Adaptive observability headers if adaptive router was used
                if hasattr(router, "last_adaptive_decision") and router.last_adaptive_decision:
                    a_dec = router.last_adaptive_decision
                    extra_response_headers["X-Selected-Model"] = str(a_dec.get("selected_model", "none"))
                    if a_dec.get("ml_predicted_server"):
                        extra_response_headers["X-ML-Predicted-Server"] = str(a_dec["ml_predicted_server"])
                    if a_dec.get("ml_confidence") is not None:
                        extra_response_headers["X-ML-Confidence"] = f"{a_dec['ml_confidence']:.4f}"
                    extra_response_headers["X-Adaptive-Fallback"] = "true" if a_dec.get("is_fallback") else "false"
                    extra_response_headers["X-Model-Selection-Time"] = f"{a_dec.get('model_selection_time_ms', 0.0):.3f}"

                # Populate default priority and deadline observability headers
                extra_response_headers["X-Request-Priority"] = priority_meta.priority.name
                req_slack = priority_meta.calculate_slack_ms(start_time)
                if req_slack is not None:
                    extra_response_headers["X-Deadline-Slack"] = f"{req_slack:.2f}"
                extra_response_headers["X-Final-Backend"] = backend
                extra_response_headers["X-Priority-Override"] = "false"
                extra_response_headers["X-Deadline-Override"] = "false"
                extra_response_headers["X-Routing-Reason"] = "standard_routing"

                # Check if PriorityDeadlineRouter was used
                if hasattr(router, "last_decision") and router.last_decision:
                    p_dec = router.last_decision
                    extra_response_headers["X-Request-Priority"] = str(p_dec.get("priority", priority_meta.priority.name))
                    if p_dec.get("slack_ms") is not None:
                        extra_response_headers["X-Deadline-Slack"] = f"{p_dec['slack_ms']:.2f}"
                    if p_dec.get("ml_predicted_server"):
                        extra_response_headers["X-ML-Predicted-Server"] = str(p_dec["ml_predicted_server"])
                    extra_response_headers["X-Final-Backend"] = str(p_dec.get("final_backend", backend))
                    extra_response_headers["X-Priority-Override"] = "true" if p_dec.get("priority_override") else "false"
                    extra_response_headers["X-Deadline-Override"] = "true" if p_dec.get("deadline_override") else "false"
                    extra_response_headers["X-Routing-Reason"] = str(p_dec.get("routing_reason", "normal"))

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
                            for h_key, h_val in extra_response_headers.items():
                                self.send_header(h_key, h_val)
                            self.end_headers()
                            self.wfile.write(body)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            pass
                        handled = True

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
                        for h_key, h_val in extra_response_headers.items():
                            self.send_header(h_key, h_val)
                        self.end_headers()
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                    handled = True

                except urllib.error.URLError as e:
                    # Mark backend down in cache
                    has_surviving = False
                    if hasattr(self.server, "_unhealthy_lock"):
                        with self.server._unhealthy_lock:
                            self.server._unhealthy_backends[backend] = time.time() + 3.0
                            active_healthy = [b for b in self.server.backends if b not in self.server._unhealthy_backends]
                            if hasattr(router, "set_healthy_backends"):
                                router.set_healthy_backends(active_healthy if active_healthy else None)
                            has_surviving = len(active_healthy) > 0

                    if attempt < max_attempts and has_surviving:
                        logger.warning("Backend %s unavailable (%s). Failing over (attempt %d)...", backend, e.reason, attempt)
                        continue

                    status_code = 502
                    self._send_json(
                        502,
                        {
                            "error": "Bad Gateway",
                            "message": f"Backend server unavailable: {e.reason}",
                            "backend": backend,
                        },
                        extra_headers=extra_response_headers,
                    )
                    handled = True

                except TimeoutError:
                    status_code = 504
                    self._send_json(
                        504,
                        {
                            "error": "Gateway Timeout",
                            "message": "Backend request timed out",
                            "backend": backend,
                        },
                        extra_headers=extra_response_headers,
                    )
                    handled = True

                except Exception as e:
                    status_code = 502
                    self._send_json(
                        502,
                        {
                            "error": "Bad Gateway",
                            "message": str(e),
                            "backend": backend,
                        },
                        extra_headers=extra_response_headers,
                    )
                    handled = True

                finally:
                    if handled:
                        end_time = time.time()
                        duration = end_time - start_time
                        pred_info = ""
                        if hasattr(router, "last_prediction") and router.last_prediction:
                            p = router.last_prediction
                            pred_info = f" ml_pred={p.get('predicted_server')} conf={p.get('confidence')} fallback={p.get('is_fallback')}"

                        logger.info(
                            "[%s] client=%s backend=%s path=%s status=%d duration=%.4fs%s",
                            algorithm_name,
                            client_ip,
                            backend,
                            self.path,
                            status_code,
                            duration,
                            pred_info,
                        )

                        # Record experimental observation if recording is active
                        if recorder and getattr(recorder, "is_recording", False) and pre_snapshot:
                            req_rate = self.headers.get("X-Request-Rate")
                            recorder.record_observation(
                                pre_snapshot=pre_snapshot,
                                experiment_id=self.headers.get("X-Experiment-ID"),
                                request_id=self.headers.get("X-Request-ID"),
                                workload_scenario=self.headers.get("X-Workload-Scenario"),
                                request_type=self.path.split("?")[0].strip("/") or "root",
                                request_size=int(self.headers.get("Content-Length", 0)),
                                concurrency=int(self.headers.get("X-Concurrency", 1)),
                                request_rate=float(req_rate) if req_rate else None,
                                routing_algorithm=algorithm_name,
                                selected_server=backend,
                                actual_response_time=round(duration * 1000.0, 3),
                                request_success=(200 <= status_code < 400),
                                request_start=start_time,
                                request_end=end_time,
                            )

    def do_POST(self):
        # Dynamic algorithm switching endpoint
        if self.path == "/lb-algorithm":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
                return

            new_algorithm = payload.get("algorithm")
            if not new_algorithm:
                self._send_json(400, {"error": "Missing 'algorithm' in payload"})
                return

            model_path = payload.get("model_path")
            adaptive_strategy = payload.get("adaptive_strategy", "policy")

            try:
                collector = getattr(self.server, "collector", None)
                backends = getattr(self.server, "backends", [])
                new_router = get_router(
                    new_algorithm,
                    backends,
                    collector=collector,
                    model_path=model_path,
                    adaptive_strategy=adaptive_strategy,
                )
                self.server.router = new_router
                self.server.algorithm = new_algorithm
                self._send_json(200, {
                    "status": "ok",
                    "algorithm": new_algorithm,
                    "router_class": new_router.__class__.__name__,
                    "model_path": model_path,
                    "adaptive_strategy": adaptive_strategy,
                })
            except Exception as e:
                self._send_json(400, {"error": f"Failed to switch algorithm: {e}"})
            return

        self._send_json(404, {"error": "Not Found"})


def create_load_balancer(
    host: Optional[str] = None,
    port: Optional[int] = None,
    algorithm: Optional[str] = None,
    backends: Optional[List[str]] = None,
    backend_timeout: Optional[float] = None,
    model_path: Optional[str] = None,
    adaptive_strategy: Optional[str] = None,
) -> ThreadingHTTPServer:
    """Create and configure a ThreadingHTTPServer instance for the load balancer."""
    resolved_host = host or os.getenv("LB_HOST", "127.0.0.1")
    resolved_port = port if port is not None else int(os.getenv("LB_PORT", "8000"))
    resolved_algorithm = algorithm or os.getenv("ROUTING_ALGORITHM", "round_robin")
    resolved_timeout = backend_timeout if backend_timeout is not None else float(os.getenv("BACKEND_TIMEOUT", "2.0"))
    resolved_model_path = model_path or os.getenv("MODEL_PATH", None)
    resolved_adaptive = adaptive_strategy or os.getenv("ADAPTIVE_STRATEGY", "meta")

    if backends is not None:
        backends_list = list(backends)
    else:
        from config.backends import get_configured_backends
        backends_list = get_configured_backends()

    collector = MetricsCollector(backends=backends_list, timeout=resolved_timeout)
    router = get_router(
        resolved_algorithm,
        backends_list,
        collector=collector,
        model_path=resolved_model_path,
        adaptive_strategy=resolved_adaptive,
    )

    server = ThreadingHTTPServer((resolved_host, resolved_port), LoadBalancerRequestHandler)
    server.router = router
    server.algorithm = resolved_algorithm
    server.backends = backends_list
    server.backend_timeout = resolved_timeout
    server.collector = collector
    server.recorder = None
    server._unhealthy_backends = {}
    server._unhealthy_lock = threading.Lock()
    return server


def run_load_balancer(
    host: Optional[str] = None,
    port: Optional[int] = None,
    algorithm: Optional[str] = None,
    backends: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    adaptive_strategy: Optional[str] = None,
):
    """Run the load balancer HTTP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    httpd = create_load_balancer(
        host=host,
        port=port,
        algorithm=algorithm,
        backends=backends,
        model_path=model_path,
        adaptive_strategy=adaptive_strategy,
    )
    logger.info(
        "Starting load balancer on %s:%d using %s with backends: %s",
        httpd.server_address[0],
        httpd.server_address[1],
        httpd.algorithm,
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
    parser.add_argument("--host", default=os.getenv("LB_HOST", "127.0.0.1"), help="Host address to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.getenv("LB_PORT", "8000")), help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--algorithm",
        default=os.getenv("ROUTING_ALGORITHM", "round_robin"),
        choices=["round_robin", "least_connections", "ip_hash", "ml", "priority_ml", "adaptive", "priority_adaptive"],
        help="Routing algorithm to use (default: round_robin)",
    )
    parser.add_argument(
        "--adaptive-strategy",
        default=os.getenv("ADAPTIVE_STRATEGY", "policy"),
        choices=["policy", "meta"],
        help="Adaptive model selection strategy (default: policy)",
    )
    parser.add_argument(
        "--backends",
        default=os.getenv("BACKENDS", None),
        help="Comma-separated list of backend URLs (e.g. http://127.0.0.1:8001,http://127.0.0.1:8002)",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", None),
        help="Path to serialized ML model artifact (joblib format)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("BACKEND_TIMEOUT", "2.0")),
        help="Backend HTTP timeout in seconds (default: 2.0)",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")] if args.backends else None
    run_load_balancer(
        host=args.host,
        port=args.port,
        algorithm=args.algorithm,
        backends=backends,
        model_path=args.model_path,
        adaptive_strategy=args.adaptive_strategy,
    )


