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

        # Route request through router (supporting priority_meta if router accepts it)
        from load_balancer.priority import PriorityDeadlineRouter
        if isinstance(router, PriorityDeadlineRouter):
            route_ctx = router.route(client_ip=client_ip, priority_meta=priority_meta)
        else:
            try:
                route_ctx = router.route(client_ip=client_ip, priority_meta=priority_meta)
            except TypeError:
                route_ctx = router.route(client_ip=client_ip)

        with route_ctx as backend:
            target_url = f"{backend.rstrip('/')}{self.path}"
            status_code = 502

            # Extract ML observability headers if ML router was used
            # Extract ML and Priority observability headers
            extra_response_headers = {"X-Backend-Server": backend}
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

            except urllib.error.URLError as e:
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

            finally:
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


def create_load_balancer(
    host: str = "127.0.0.1",
    port: int = 8000,
    algorithm: str = "round_robin",
    backends: Optional[List[str]] = None,
    backend_timeout: float = 2.0,
    model_path: Optional[str] = None,
    adaptive_strategy: str = "policy",
) -> ThreadingHTTPServer:
    """Create and configure a ThreadingHTTPServer instance for the load balancer."""
    backends_list = list(backends) if backends else list(DEFAULT_BACKENDS)
    collector = MetricsCollector(backends=backends_list, timeout=backend_timeout)
    router = get_router(
        algorithm,
        backends_list,
        collector=collector,
        model_path=model_path,
        adaptive_strategy=adaptive_strategy,
    )

    server = ThreadingHTTPServer((host, port), LoadBalancerRequestHandler)
    server.router = router
    server.algorithm = algorithm
    server.backends = backends_list
    server.backend_timeout = backend_timeout
    server.collector = collector
    server.recorder = None
    return server


def run_load_balancer(
    host: str = "127.0.0.1",
    port: int = 8000,
    algorithm: str = "round_robin",
    backends: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    adaptive_strategy: str = "policy",
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
        choices=["round_robin", "least_connections", "ip_hash", "ml", "priority_ml", "adaptive", "priority_adaptive"],
        help="Routing algorithm to use (default: round_robin)",
    )
    parser.add_argument(
        "--adaptive-strategy",
        default="policy",
        choices=["policy", "meta"],
        help="Adaptive model selection strategy (default: policy)",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated list of backend URLs (e.g. http://127.0.0.1:8001,http://127.0.0.1:8002)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to serialized ML model artifact (joblib format)",
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


