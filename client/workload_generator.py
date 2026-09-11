"""Controlled Workload Generator for local HTTP load balancing experiments."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import logging
import math
import random
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

logger = logging.getLogger("workload_generator")


@dataclass
class RequestRecord:
    """Record of an individual executed HTTP request."""

    request_id: int
    timestamp: float
    target: str
    request_type: str
    request_duration: float
    request_start: float
    request_end: float
    success: bool
    status_code: int
    response_time: float  # in milliseconds
    backend_server: Optional[str] = None
    error: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class WorkloadSummary:
    """Aggregated metrics summary of a completed workload run."""

    scenario_name: str
    target_url: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_sec: float
    actual_throughput_rps: float
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadConfig:
    """Configuration specification for a controlled workload execution."""

    target_url: str = "http://127.0.0.1:8000"
    num_requests: int = 30
    concurrency: int = 5
    request_rate: Optional[float] = None       # Target requests per second (None = unthrottled)
    endpoint: str = "/process"                 # "/process", "/health", or "mixed"
    request_duration: float = 0.05             # Parameter passed to /process?duration=...
    burst_size: Optional[int] = None           # Requests per burst (for burst scenarios)
    burst_interval: Optional[float] = None     # Pause between bursts in seconds
    seed: Optional[int] = 42                   # Random seed for reproducible runs
    scenario_name: str = "custom"
    experiment_id: Optional[str] = None        # Associated experiment identifier
    priority_profile: Optional[str] = None     # E.g. 'equal', 'mixed', 'tight_deadlines', 'conflict'
    default_priority: Optional[str] = None     # E.g. 'HIGH', 'CRITICAL'
    default_deadline_offset: Optional[float] = None  # E.g. 0.2 seconds relative to dispatch


    def validate(self) -> None:
        """Validate configuration values."""
        if not self.target_url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid target_url: '{self.target_url}'. Must start with http:// or https://")
        if self.num_requests <= 0:
            raise ValueError(f"num_requests must be positive, got {self.num_requests}")
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {self.concurrency}")
        if self.request_duration < 0:
            raise ValueError(f"request_duration must be non-negative, got {self.request_duration}")
        if self.request_rate is not None and self.request_rate <= 0:
            raise ValueError(f"request_rate must be positive, got {self.request_rate}")
        if self.burst_size is not None and self.burst_size <= 0:
            raise ValueError(f"burst_size must be positive, got {self.burst_size}")
        if self.burst_interval is not None and self.burst_interval < 0:
            raise ValueError(f"burst_interval must be non-negative, got {self.burst_interval}")


# Pre-configured reproducible workload scenarios
SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # Phase 16 Calibrated Regimes
    "stable_normal": {
        "num_requests": 30,
        "concurrency": 2,
        "request_rate": 8.0,
        "endpoint": "/process",
        "request_duration": 0.015,
        "priority_profile": "equal",
    },
    "dynamic_moderate": {
        "num_requests": 40,
        "concurrency": 5,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.035,
        "priority_profile": "equal",
    },
    "burst_spike": {
        "num_requests": 50,
        "concurrency": 10,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.045,
        "burst_size": 15,
        "burst_interval": 0.20,
        "priority_profile": "equal",
    },
    "sustained_stress": {
        "num_requests": 60,
        "concurrency": 14,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.080,
        "priority_profile": "equal",
    },
    "priority_conflict": {
        "num_requests": 45,
        "concurrency": 8,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.040,
        "priority_profile": "conflict",
    },
    "adaptive_multiphase": {
        "num_requests": 75,
        "concurrency": 8,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.040,
        "priority_profile": "mixed",
    },
    # Backward Compatibility & Legacy Templates
    "steady_state": {
        "num_requests": 30,
        "concurrency": 5,
        "request_rate": 10.0,
        "endpoint": "/process",
        "request_duration": 0.03,
        "priority_profile": "equal",
    },
    "stress_overload": {
        "num_requests": 60,
        "concurrency": 12,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.06,
        "priority_profile": "equal",
    },
    "mixed_priority": {
        "num_requests": 40,
        "concurrency": 6,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.03,
        "priority_profile": "mixed",
    },
    "high_contention_conflict": {
        "num_requests": 40,
        "concurrency": 8,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.04,
        "priority_profile": "conflict",
    },
    "low_traffic": {
        "num_requests": 20,
        "concurrency": 2,
        "request_rate": 5.0,
        "endpoint": "/process",
        "request_duration": 0.02,
    },
    "medium_traffic": {
        "num_requests": 50,
        "concurrency": 5,
        "request_rate": 15.0,
        "endpoint": "/process",
        "request_duration": 0.03,
    },
    "high_traffic": {
        "num_requests": 100,
        "concurrency": 15,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.03,
    },
    "burst_traffic": {
        "num_requests": 60,
        "concurrency": 20,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.04,
        "burst_size": 20,
        "burst_interval": 0.3,
    },
    "cpu_heavy": {
        "num_requests": 30,
        "concurrency": 5,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.12,
    },
    "mixed": {
        "num_requests": 60,
        "concurrency": 6,
        "request_rate": 20.0,
        "endpoint": "mixed",
        "request_duration": 0.04,
    },
    "dynamic": {
        "num_requests": 80,
        "concurrency": 10,
        "request_rate": None,
        "endpoint": "/process",
        "request_duration": 0.03,
    },
}


def get_scenario(
    name: str,
    target_url: str = "http://127.0.0.1:8000",
    **overrides: Any,
) -> WorkloadConfig:
    """Instantiate a configured WorkloadConfig based on a named scenario template."""
    normalized = name.lower().replace("-", "_").strip()
    if normalized not in SCENARIO_TEMPLATES:
        valid = list(SCENARIO_TEMPLATES.keys())
        raise ValueError(f"Unknown scenario '{name}'. Supported scenarios: {valid}")

    params = dict(SCENARIO_TEMPLATES[normalized])
    params.update(overrides)
    params["scenario_name"] = normalized
    params["target_url"] = target_url

    config = WorkloadConfig(**params)
    config.validate()
    return config


class WorkloadGenerator:
    """Executes controlled, reproducible HTTP traffic against targets."""

    def __init__(self, config: WorkloadConfig, stop_event: Optional[threading.Event] = None):
        config.validate()
        self.config = config
        self.stop_event = stop_event
        self._rng = random.Random(config.seed)
        self._records: List[RequestRecord] = []
        self._lock = threading.Lock()

    def _build_request_item(self, request_id: int) -> tuple[str, str, float]:
        """Construct (url, request_type, request_duration) for a request."""
        endpoint = self.config.endpoint
        base = self.config.target_url.rstrip("/")

        if endpoint == "mixed":
            # Deterministic pseudo-random selection based on seeded RNG
            choice = self._rng.choice(["health", "process_light", "process_heavy"])
            if choice == "health":
                return f"{base}/health", "health", 0.0
            elif choice == "process_light":
                duration = 0.02
                return f"{base}/process?duration={duration}", "process", duration
            else:
                duration = 0.08
                return f"{base}/process?duration={duration}", "process", duration
        elif endpoint == "/health":
            return f"{base}/health", "health", 0.0
        elif self.config.scenario_name == "adaptive_multiphase":
            # 5-stage progression across num_requests:
            # Stage 1 (0-20%): Low Load / Normal (duration = 0.015s)
            # Stage 2 (20-40%): Moderate Load (duration = 0.035s)
            # Stage 3 (40-60%): Burst Spike (duration = 0.045s)
            # Stage 4 (60-80%): Stress Overload (duration = 0.080s)
            # Stage 5 (80-100%): Recovery (duration = 0.015s)
            fraction = (request_id - 1) / max(1, self.config.num_requests)
            if fraction < 0.20:
                duration = 0.015
            elif fraction < 0.40:
                duration = 0.035
            elif fraction < 0.60:
                duration = 0.045
            elif fraction < 0.80:
                duration = 0.080
            else:
                duration = 0.015
            return f"{base}/process?duration={duration}", "process", duration
        else:
            duration = self.config.request_duration
            return f"{base}/process?duration={duration}", "process", duration

    def _execute_single_request(self, request_id: int, url: str, req_type: str, req_duration: float) -> RequestRecord:
        """Execute a single HTTP request and record its latency and metadata."""
        if self.stop_event and self.stop_event.is_set():
            t_now = time.time()
            return RequestRecord(
                request_id=request_id,
                timestamp=t_now,
                target=url,
                request_type=req_type,
                request_duration=req_duration,
                request_start=time.perf_counter(),
                request_end=time.perf_counter(),
                success=False,
                status_code=499,
                response_time=0.0,
                error="Cancelled by user stop_event",
            )

        start_perf = time.perf_counter()
        timestamp = time.time()
        success = False
        status_code = 0
        backend_server = None
        error_msg = None

        try:
            req_headers = {
                "User-Agent": "WorkloadGenerator/1.0",
                "X-Request-ID": str(request_id),
                "X-Workload-Scenario": str(self.config.scenario_name),
                "X-Concurrency": str(self.config.concurrency),
                "X-Arrival-Time": str(timestamp),
                "X-Estimated-Duration": str(req_duration),
            }
            if self.config.experiment_id:
                req_headers["X-Experiment-ID"] = str(self.config.experiment_id)
            if self.config.request_rate:
                req_headers["X-Request-Rate"] = str(self.config.request_rate)

            # Priority & Deadline generation based on priority profile
            prof = self.config.priority_profile
            p_val = self.config.default_priority or "NORMAL"
            d_val = None

            if prof == "equal":
                p_val = "NORMAL"
                d_val = None
            elif prof == "mixed":
                # Stochastically distribute across LOW, NORMAL, HIGH, CRITICAL
                p_val = self._rng.choice(["LOW", "NORMAL", "HIGH", "CRITICAL"])
                if p_val in ("HIGH", "CRITICAL"):
                    d_val = f"+{round(self._rng.uniform(0.15, 0.4), 2)}"
            elif prof == "tight_deadlines":
                p_val = self._rng.choice(["NORMAL", "HIGH", "CRITICAL"])
                d_val = f"+{round(self._rng.uniform(0.06, 0.15), 2)}"
            elif prof == "conflict":
                # Interleaved conflict pattern:
                # Odd requests: HIGH priority but relaxed deadline (+0.6s)
                # Even requests: LOW priority but urgent deadline (+0.08s)
                if request_id % 2 == 1:
                    p_val = "HIGH"
                    d_val = "+0.60"
                else:
                    p_val = "LOW"
                    d_val = "+0.08"
            elif self.config.default_deadline_offset is not None:
                d_val = f"+{self.config.default_deadline_offset:.2f}"

            req_headers["X-Request-Priority"] = p_val
            if d_val:
                req_headers["X-Request-Deadline"] = d_val

            req = urllib.request.Request(
                url=url,
                headers=req_headers,
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status_code = resp.status
                success = (200 <= status_code < 400)
                backend_server = resp.headers.get("X-Backend-Server")
                resp_headers = dict(resp.headers)
                # Also try to parse server_id from body if present
                try:
                    body_json = json.loads(resp.read().decode("utf-8"))
                    if not backend_server and "server_id" in body_json:
                        backend_server = body_json["server_id"]
                except Exception:
                    pass

        except urllib.error.HTTPError as e:
            status_code = e.code
            success = False
            error_msg = f"HTTP {e.code}: {e.reason}"
            backend_server = e.headers.get("X-Backend-Server")
            resp_headers = dict(e.headers) if hasattr(e, "headers") and e.headers else None

        except urllib.error.URLError as e:
            status_code = 0
            success = False
            error_msg = f"Connection failed: {e.reason}"
            resp_headers = None

        except TimeoutError:
            status_code = 0
            success = False
            error_msg = "Request timed out"
            resp_headers = None

        except Exception as e:
            status_code = 0
            success = False
            error_msg = str(e)
            resp_headers = None

        end_perf = time.perf_counter()
        response_time_ms = round((end_perf - start_perf) * 1000.0, 3)

        record = RequestRecord(
            request_id=request_id,
            timestamp=timestamp,
            target=url,
            request_type=req_type,
            request_duration=req_duration,
            request_start=start_perf,
            request_end=end_perf,
            success=success,
            status_code=status_code,
            response_time=response_time_ms,
            backend_server=backend_server,
            error=error_msg,
            response_headers=resp_headers,
        )


        with self._lock:
            self._records.append(record)

        return record

    def run(self) -> List[RequestRecord]:
        """Run the configured workload and return records for all executed requests."""
        self._records = []
        total_requests = self.config.num_requests
        rate = self.config.request_rate
        burst_size = self.config.burst_size
        burst_interval = self.config.burst_interval or 0.0

        # Pre-plan requests using RNG for reproducibility
        request_items = [self._build_request_item(i) for i in range(1, total_requests + 1)]

        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = []

            if burst_size:
                # Burst dispatch mode
                for i in range(0, total_requests, burst_size):
                    if self.stop_event and self.stop_event.is_set():
                        break
                    batch = request_items[i : i + burst_size]
                    for idx, (url, rtype, rduration) in enumerate(batch, start=i + 1):
                        futures.append(
                            executor.submit(self._execute_single_request, idx, url, rtype, rduration)
                        )
                    if i + burst_size < total_requests and burst_interval > 0:
                        time.sleep(burst_interval)
            elif rate and rate > 0:
                # Rate-controlled dispatch mode
                delay = 1.0 / rate
                for req_id, (url, rtype, rduration) in enumerate(request_items, start=1):
                    if self.stop_event and self.stop_event.is_set():
                        break
                    futures.append(
                        executor.submit(self._execute_single_request, req_id, url, rtype, rduration)
                    )
                    time.sleep(delay)
            else:
                # Concurrent unthrottled dispatch
                for req_id, (url, rtype, rduration) in enumerate(request_items, start=1):
                    if self.stop_event and self.stop_event.is_set():
                        break
                    futures.append(
                        executor.submit(self._execute_single_request, req_id, url, rtype, rduration)
                    )

            # Await all completions
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error("Error in request task: %s", e)

        # Sort records deterministically by request_id
        self._records.sort(key=lambda r: r.request_id)
        return list(self._records)

    def get_summary(self) -> WorkloadSummary:
        """Compute summary statistics for the completed workload."""
        if not self._records:
            return WorkloadSummary(
                scenario_name=self.config.scenario_name,
                target_url=self.config.target_url,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_duration_sec=0.0,
                actual_throughput_rps=0.0,
                avg_response_time_ms=0.0,
                p50_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
            )

        total = len(self._records)
        successes = sum(1 for r in self._records if r.success)
        failures = total - successes

        earliest_start = min(r.request_start for r in self._records)
        latest_end = max(r.request_end for r in self._records)
        total_duration = max(0.001, latest_end - earliest_start)
        throughput = round(total / total_duration, 2)

        latencies = sorted(r.response_time for r in self._records)

        def percentile(data: List[float], p: float) -> float:
            k = (len(data) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return round(data[int(k)], 3)
            return round(data[f] * (c - k) + data[c] * (k - f), 3)

        return WorkloadSummary(
            scenario_name=self.config.scenario_name,
            target_url=self.config.target_url,
            total_requests=total,
            successful_requests=successes,
            failed_requests=failures,
            total_duration_sec=round(total_duration, 3),
            actual_throughput_rps=throughput,
            avg_response_time_ms=round(sum(latencies) / total, 3),
            p50_response_time_ms=percentile(latencies, 0.50),
            p95_response_time_ms=percentile(latencies, 0.95),
            p99_response_time_ms=percentile(latencies, 0.99),
            min_response_time_ms=latencies[0],
            max_response_time_ms=latencies[-1],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Controlled HTTP Workload Generator")
    parser.add_argument("--scenario", default="low_traffic", choices=list(SCENARIO_TEMPLATES.keys()), help="Scenario template")
    parser.add_argument("--target", default="http://127.0.0.1:8000", help="Target URL (e.g. Load Balancer or Server)")
    parser.add_argument("--requests", type=int, default=None, help="Override total requests")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency level")
    parser.add_argument("--rate", type=float, default=None, help="Override request rate (req/s)")
    parser.add_argument("--duration", type=float, default=None, help="Override /process request duration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default=None, help="Optional JSON file to save detailed records")
    args = parser.parse_args()

    overrides = {}
    if args.requests is not None:
        overrides["num_requests"] = args.requests
    if args.concurrency is not None:
        overrides["concurrency"] = args.concurrency
    if args.rate is not None:
        overrides["request_rate"] = args.rate
    if args.duration is not None:
        overrides["request_duration"] = args.duration
    if args.seed is not None:
        overrides["seed"] = args.seed

    cfg = get_scenario(args.scenario, target_url=args.target, **overrides)
    gen = WorkloadGenerator(cfg)
    print(f"Running scenario '{cfg.scenario_name}' against {cfg.target_url}...")
    records = gen.run()
    summary = gen.get_summary()
    print("Workload Summary:")
    print(json.dumps(summary.to_dict(), indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        print(f"Saved {len(records)} records to {args.output}")
