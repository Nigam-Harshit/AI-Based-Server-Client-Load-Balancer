import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("server")


class ServerRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for server node."""

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

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == "/health":
            self._send_json(200, {
                "status": "ok",
                "server_id": self.server_id,
                "port": self.server_port,
            })
        elif path == "/process":
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
        else:
            self._send_json(404, {
                "error": "Not Found",
                "path": path,
            })


def create_server(server_id: str, host: str = "127.0.0.1", port: int = 8001) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance."""
    server = ThreadingHTTPServer((host, port), ServerRequestHandler)
    server.server_id = server_id
    server.server_port = port
    return server


def run_server(server_id: str, host: str = "127.0.0.1", port: int = 8001):
    """Run the server node indefinitely."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    httpd = create_server(server_id, host, port)
    logger.info("Starting server %s on %s:%d", server_id, host, port)
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
    args = parser.parse_args()

    run_server(server_id=args.id, host=args.host, port=args.port)
