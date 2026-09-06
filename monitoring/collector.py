"""Real-time metric collection abstraction for backend servers."""

from dataclasses import asdict, dataclass
import json
import logging
import time
from typing import Dict, List, Optional
import urllib.error
import urllib.request

from config.backends import DEFAULT_BACKENDS

logger = logging.getLogger("monitoring")


@dataclass
class ServerMetrics:
    """Represents real measured runtime metrics for a single backend server."""

    server_id: str
    backend_url: str
    cpu_percent: float          # Process CPU utilization (0.0 to 100.0%)
    memory_percent: float       # Process Memory utilization (0.0 to 100.0%)
    active_connections: int     # Number of requests currently being processed
    avg_response_time_ms: float # Average duration of recent requests in ms
    network_latency_ms: float   # Round-trip probe latency between collector and server in ms
    request_queue_length: int   # Number of requests waiting in queue for execution
    timestamp: float            # Measurement epoch timestamp
    available: bool = True      # Whether backend responded successfully
    error: Optional[str] = None # Error details if unavailable

    def to_dict(self) -> dict:
        return asdict(self)


class MetricsCollector:
    """Collects real-time metrics across all configured backend servers."""

    def __init__(self, backends: Optional[List[str]] = None, timeout: float = 2.0):
        self.backends = list(backends) if backends else list(DEFAULT_BACKENDS)
        self.timeout = timeout

    def collect_from_server(self, backend_url: str) -> ServerMetrics:
        """Query /metrics on a single backend server and measure network latency."""
        url = f"{backend_url.rstrip('/')}/metrics"
        start_time = time.perf_counter()

        try:
            req = urllib.request.Request(
                url=url,
                headers={"User-Agent": "MetricsCollector/1.0"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                elapsed_rtt_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
                data = json.loads(resp.read().decode("utf-8"))

                return ServerMetrics(
                    server_id=data.get("server_id", "unknown"),
                    backend_url=backend_url,
                    cpu_percent=float(data.get("cpu_percent", 0.0)),
                    memory_percent=float(data.get("memory_percent", 0.0)),
                    active_connections=int(data.get("active_connections", 0)),
                    avg_response_time_ms=float(data.get("avg_response_time_ms", 0.0)),
                    # Measured round-trip probe latency from collector to server
                    network_latency_ms=elapsed_rtt_ms,
                    request_queue_length=int(data.get("request_queue_length", 0)),
                    timestamp=time.time(),
                    available=True,
                )

        except urllib.error.HTTPError as e:
            logger.warning("HTTP error %d querying metrics from %s", e.code, backend_url)
            return ServerMetrics(
                server_id="unknown",
                backend_url=backend_url,
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                avg_response_time_ms=0.0,
                network_latency_ms=0.0,
                request_queue_length=0,
                timestamp=time.time(),
                available=False,
                error=f"HTTP {e.code}: {e.reason}",
            )

        except urllib.error.URLError as e:
            logger.warning("Network error querying metrics from %s: %s", backend_url, e.reason)
            return ServerMetrics(
                server_id="unknown",
                backend_url=backend_url,
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                avg_response_time_ms=0.0,
                network_latency_ms=0.0,
                request_queue_length=0,
                timestamp=time.time(),
                available=False,
                error=f"Unreachable: {e.reason}",
            )

        except TimeoutError:
            logger.warning("Timeout querying metrics from %s", backend_url)
            return ServerMetrics(
                server_id="unknown",
                backend_url=backend_url,
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                avg_response_time_ms=0.0,
                network_latency_ms=0.0,
                request_queue_length=0,
                timestamp=time.time(),
                available=False,
                error="Timeout",
            )

        except Exception as e:
            logger.warning("Unexpected error querying metrics from %s: %s", backend_url, e)
            return ServerMetrics(
                server_id="unknown",
                backend_url=backend_url,
                cpu_percent=0.0,
                memory_percent=0.0,
                active_connections=0,
                avg_response_time_ms=0.0,
                network_latency_ms=0.0,
                request_queue_length=0,
                timestamp=time.time(),
                available=False,
                error=str(e),
            )

    def collect_all(self) -> Dict[str, ServerMetrics]:
        """Collect metrics from all configured backend servers."""
        results: Dict[str, ServerMetrics] = {}
        for backend in self.backends:
            results[backend] = self.collect_from_server(backend)
        return results

    def get_summary(self) -> dict:
        """Return dictionary summary of all current backend metrics."""
        return {b: m.to_dict() for b, m in self.collect_all().items()}
