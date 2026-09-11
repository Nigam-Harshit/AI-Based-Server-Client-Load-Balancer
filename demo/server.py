"""HTTP Server for Phase 16 Real-Time Demonstration & Observability Console."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
import sys
from typing import Any, Dict, Optional
import urllib.parse

from demo.cluster_manager import ClusterManager

logger = logging.getLogger("demo.server")


class DemoRequestHandler(BaseHTTPRequestHandler):
    """Handles static web UI assets and REST API requests for the demo console."""

    cluster_mgr: ClusterManager = None
    ui_dir: str = os.path.join(os.path.dirname(__file__), "ui")

    def log_message(self, format, *args):
        # Suppress noisy access logs; log warnings/errors
        pass

    def _send_json(self, status_code: int, data: Any):
        try:
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _send_text(self, status_code: int, content: str, content_type: str = "text/plain"):
        try:
            body = content.encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _parse_body(self) -> Dict[str, Any]:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len <= 0:
            return {}
        raw = self.rfile.read(content_len).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_OPTIONS(self):
        """Handle CORS pre-flight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # REST API endpoints
        if path == "/api/status":
            self._send_json(200, self.cluster_mgr.get_cluster_status())
            return

        if path == "/api/telemetry":
            self._send_json(200, self.cluster_mgr.get_telemetry())
            return

        if path == "/api/scenarios":
            scenarios_file = os.path.join(self.cluster_mgr.data_dir, "scenarios", "preset_scenarios.json")
            if os.path.exists(scenarios_file):
                with open(scenarios_file, "r", encoding="utf-8") as f:
                    self._send_json(200, json.load(f))
            else:
                self._send_json(200, [])
            return

        if path == "/api/demo/status":
            self._send_json(200, self.cluster_mgr.get_guided_demo_state())
            return

        if path == "/api/export":
            query = urllib.parse.parse_qs(parsed.query)
            fmt = query.get("format", ["json"])[0]
            data = self.cluster_mgr.export_data(format=fmt)
            c_type = "text/csv" if fmt == "csv" else "application/json"
            self._send_text(200, data, content_type=c_type)
            return

        # Static UI asset serving
        if path in ("/", "/index.html", "/ui", "/ui/"):
            target_path = os.path.join(self.ui_dir, "index.html")
        elif path.startswith("/ui/"):
            rel_path = path[4:].lstrip("/")
            target_path = os.path.join(self.ui_dir, rel_path)
        else:
            target_path = os.path.join(self.ui_dir, path.lstrip("/"))

        if os.path.isfile(target_path):
            mime_type, _ = mimetypes.guess_type(target_path)
            mime_type = mime_type or "application/octet-stream"
            try:
                with open(target_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", f"{mime_type}; charset=utf-8" if "text" in mime_type or "javascript" in mime_type else mime_type)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self._send_json(500, {"error": f"Failed reading asset: {e}"})
        else:
            self._send_json(404, {"error": f"File not found: {path}"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        body = self._parse_body()

        if path == "/api/algorithm":
            algo = body.get("algorithm")
            if not algo:
                self._send_json(400, {"error": "Missing 'algorithm' parameter"})
                return
            model_path = body.get("model_path")
            adaptive_strategy = body.get("adaptive_strategy", "policy")
            adaptive_strategy = body.get("adaptive_strategy", "meta")
            try:
                res = self.cluster_mgr.switch_algorithm(algo, model_path=model_path, adaptive_strategy=adaptive_strategy)
                self._send_json(200, res)
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        if path == "/api/workload/start":
            try:
                res = self.cluster_mgr.start_workload(body)
                self._send_json(200, res)
            except Exception as e:
                self._send_json(400, {"error": str(e)})
            return

        if path == "/api/workload/stop":
            res = self.cluster_mgr.stop_workload()
            self._send_json(200, res)
            return

        if path == "/api/backend/toggle":
            b_url = body.get("backend")
            action = body.get("action", "toggle")
            if not b_url:
                self._send_json(400, {"error": "Missing 'backend' parameter"})
                return
            res = self.cluster_mgr.toggle_backend(b_url, action=action)
            self._send_json(200, res)
            return

        if path == "/api/cluster/ensure":
            status = self.cluster_mgr.ensure_local_cluster()
            self._send_json(200, {"status": "ok", "backends": status})
            return

        if path == "/api/session/start":
            name = body.get("name", "Demo Session")
            desc = body.get("description", "")
            res = self.cluster_mgr.start_session(name, desc)
            self._send_json(200, res)
            return

        if path == "/api/session/end":
            res = self.cluster_mgr.end_session()
            self._send_json(200, res)
            return

        if path == "/api/demo/run":
            res = self.cluster_mgr.start_guided_demo()
            self._send_json(200, res)
            return

        self._send_json(404, {"error": f"Endpoint not found: {path}"})


def create_demo_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    lb_url: str = "http://127.0.0.1:8000",
    backends: Optional[list[str]] = None,
) -> ThreadingHTTPServer:
    """Instantiate and configure ThreadingHTTPServer for the demo console."""
    cluster_mgr = ClusterManager(lb_url=lb_url, backends=backends)
    DemoRequestHandler.cluster_mgr = cluster_mgr
    server = ThreadingHTTPServer((host, port), DemoRequestHandler)
    server.cluster_mgr = cluster_mgr
    return server


def run_demo_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    lb_url: str = "http://127.0.0.1:8000",
    ensure_cluster: bool = True,
):
    """Launch the demo console server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    srv = create_demo_server(host=host, port=port, lb_url=lb_url)

    if ensure_cluster:
        logger.info("Verifying and starting local cluster services...")
        srv.cluster_mgr.ensure_local_cluster()

    logger.info("==================================================================")
    logger.info(" Phase 16 Real-Time Demonstration & Observability Console Active")
    logger.info(" Web UI available at: http://%s:%d", host, port)
    logger.info(" Load Balancer Target: %s", lb_url)
    logger.info("==================================================================")

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Demo Console server...")
    finally:
        srv.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 16 Demonstration Console Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Demo Console port (default: 8080)")
    parser.add_argument("--lb-url", default="http://127.0.0.1:8000", help="Load Balancer URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--no-auto-cluster", action="store_true", help="Do not automatically start background backends")
    args = parser.parse_args()

    run_demo_server(
        host=args.host,
        port=args.port,
        lb_url=args.lb_url,
        ensure_cluster=not args.no_auto_cluster,
    )

