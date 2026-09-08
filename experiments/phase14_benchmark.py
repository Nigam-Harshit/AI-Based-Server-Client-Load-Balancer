"""Phase 14: End-to-End System Benchmarking and Final Performance Validation.

Executes a comprehensive, controlled, and matched empirical benchmark of:
- 10 routing strategies:
  1. Round Robin
  2. Least Connections
  3. IP Hash
  4. Logistic Regression
  5. Random Forest
  6. Decision Tree
  7. SVM
  8. XGBoost
  9. Adaptive Policy Selector
  10. Adaptive Meta-Selector
- 9 operational regimes:
  low_load, medium_load, high_load, burst_load, cpu_heavy,
  mixed_workload, dynamic_workload, queue_contention, backend_imbalance.
- Matched repetitions (R >= 5) with synchronized random seeds.
- Separate isolated evaluation of Priority and Deadline-Aware routing.
- Accurate, neutral, non-negative measurement of routing decision overhead.
- Generates data/phase14/phase14_raw.csv, experiment_manifest.json, environment.json.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import logging
import os
import platform
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request
import psutil

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.backends import DEFAULT_BACKENDS
from load_balancer.app import create_load_balancer
from load_balancer.router import (
    BaseRouter,
    RoundRobinRouter,
    LeastConnectionsRouter,
    IPHashRouter,
    MLRouter,
    get_router,
)
from ml.adaptive_selector import AdaptiveRouter
from monitoring.collector import MetricsCollector, ServerMetrics
from server.app import create_server

logger = logging.getLogger("experiments.phase14_benchmark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "phase14")
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "phase14_raw.csv")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "experiment_manifest.json")
ENV_PATH = os.path.join(OUTPUT_DIR, "environment.json")

ALGORITHMS = [
    "round_robin",
    "least_connections",
    "ip_hash",
    "logistic_regression",
    "random_forest",
    "decision_tree",
    "svm",
    "xgboost",
    "adaptive_policy",
    "adaptive_meta",
]

REGIMES = [
    "low_load",
    "medium_load",
    "high_load",
    "burst_load",
    "cpu_heavy",
    "mixed_workload",
    "dynamic_workload",
    "queue_contention",
    "backend_imbalance",
]

MODEL_PATH_MAP = {
    "logistic_regression": "models/logistic_regression.joblib",
    "random_forest": "models/random_forest.joblib",
    "decision_tree": "models/decision_tree.joblib",
    "svm": "models/svm.joblib",
    "xgboost": "models/xgboost.joblib",
}

RAW_FIELDNAMES = [
    "experiment_id",
    "run_idx",
    "algorithm",
    "regime",
    "request_id",
    "timestamp",
    "request_type",
    "request_duration",
    "concurrency",
    "request_rate",
    "seed",
    "backend_server",
    "success",
    "status_code",
    "response_time_ms",
    "routing_overhead_ms",
    "ml_predicted_server",
    "ml_confidence",
    "ml_fallback",
    "selected_model",
    "model_selection_time_ms",
    "s1_cpu_pre",
    "s1_mem_pre",
    "s1_conn_pre",
    "s1_resp_pre",
    "s1_lat_pre",
    "s2_cpu_pre",
    "s2_mem_pre",
    "s2_conn_pre",
    "s2_resp_pre",
    "s2_lat_pre",
    "s3_cpu_pre",
    "s3_mem_pre",
    "s3_conn_pre",
    "s3_resp_pre",
    "s3_lat_pre",
    "optimal_server",
    "suboptimal_routing",
    "is_priority_subset",
    "priority",
    "deadline_ms",
    "deadline_met",
]


class BenchmarkCluster:
    """Manages 3 backend HTTP servers and 1 central HTTP Load Balancer."""

    def __init__(
        self,
        lb_port: int = 8000,
        server_ports: Optional[List[int]] = None,
        backend_timeout: float = 2.0,
    ):
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.backends = [f"http://127.0.0.1:{p}" for p in self.server_ports]
        self.backend_timeout = backend_timeout
        self.servers = []
        self.server_threads = []
        self.lb = None
        self.lb_thread = None
        self.collector = MetricsCollector(backends=self.backends)

    def start(self) -> None:
        """Start all 3 backend servers and load balancer."""
        for i, port in enumerate(self.server_ports, start=1):
            srv = create_server(server_id=f"server-{i}", host="127.0.0.1", port=port)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            self.servers.append(srv)
            self.server_threads.append(t)

        for port in self.server_ports:
            self._wait_for_url(f"http://127.0.0.1:{port}/health")

        # Initialize load balancer with initial round-robin router
        self.lb = create_load_balancer(
            host="127.0.0.1",
            port=self.lb_port,
            algorithm="round_robin",
            backends=self.backends,
            backend_timeout=self.backend_timeout,
        )
        self.lb_thread = threading.Thread(target=self.lb.serve_forever, daemon=True)
        self.lb_thread.start()
        self._wait_for_url(f"http://127.0.0.1:{self.lb_port}/health")
        logger.info("Benchmark cluster online: LB on :%d, Backends on %s", self.lb_port, self.server_ports)

    def set_router(self, algorithm: str, priority: bool = False) -> None:
        """Dynamically configure the load balancer router for the target algorithm."""
        norm = algorithm.lower().strip()
        if priority:
            if norm == "round_robin":
                base = RoundRobinRouter(self.backends)
            elif norm == "least_connections":
                base = LeastConnectionsRouter(self.backends)
            elif norm.startswith("ml") or norm == "logistic_regression":
                base = MLRouter(self.backends, collector=self.collector, model_path="models/logistic_regression.joblib")
            elif norm.startswith("adaptive"):
                base = AdaptiveRouter(self.backends, collector=self.collector, strategy="policy")
            else:
                base = LeastConnectionsRouter(self.backends)
            from load_balancer.priority import PriorityDeadlineRouter
            router = PriorityDeadlineRouter(base_router=base, collector=self.collector, backends=self.backends)
        elif norm in ("round_robin", "least_connections", "ip_hash"):
            router = get_router(norm, self.backends)
        elif norm in MODEL_PATH_MAP:
            model_path = os.path.join(PROJECT_ROOT, MODEL_PATH_MAP[norm])
            router = MLRouter(self.backends, collector=self.collector, model_path=model_path)
        elif norm == "adaptive_policy":
            router = AdaptiveRouter(self.backends, collector=self.collector, strategy="policy")
        elif norm == "adaptive_meta":
            router = AdaptiveRouter(self.backends, collector=self.collector, strategy="meta")
        else:
            router = get_router("round_robin", self.backends)

        self.lb.router = router
        self.lb.algorithm = algorithm

    def stop(self) -> None:
        """Stop load balancer and backend servers."""
        if self.lb:
            try:
                self.lb.shutdown()
                self.lb.server_close()
            except Exception:
                pass
        for srv in self.servers:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
        logger.info("Benchmark cluster stopped cleanly.")

    def _wait_for_url(self, url: str, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"Service at {url} did not respond within {timeout}s")


def calculate_optimal_server(snapshot: Dict[str, Any]) -> str:
    """Compute empirical optimal backend using pre-routing cluster cost formula:
    cost_i = latency_i + resp_time_i * (1 + connections_i)
    """
    costs = []
    for i in range(1, 4):
        lat = float(snapshot.get(f"s{i}_lat_pre", 0.0))
        resp = float(snapshot.get(f"s{i}_resp_pre", 0.0))
        conn = int(snapshot.get(f"s{i}_conn_pre", 0))
        cost = lat + resp * (1.0 + conn)
        costs.append((f"server-{i}", cost))
    return min(costs, key=lambda x: x[1])[0]


def start_backend_stress(backend_url: str, duration: float = 0.08, stop_event: Optional[threading.Event] = None):
    """Run real continuous CPU stress against a specific backend until stopped."""
    def _stress():
        while stop_event is None or not stop_event.is_set():
            try:
                url = f"{backend_url.rstrip('/')}/process?duration={duration}"
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    resp.read()
            except Exception:
                pass
            time.sleep(0.005)

    t = threading.Thread(target=_stress, daemon=True)
    t.start()
    return t


class Phase14BenchmarkRunner:
    """Orchestrates the Phase 14 experimental benchmark suite."""

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        repetitions: int = 5,
        base_seeds: Optional[List[int]] = None,
        lb_port: int = 8000,
        server_ports: Optional[List[int]] = None,
    ):
        self.output_dir = output_dir
        self.repetitions = repetitions
        self.base_seeds = base_seeds or [42, 43, 44, 45, 46][:repetitions]
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.cluster = BenchmarkCluster(lb_port=lb_port, server_ports=server_ports)
        self.global_request_counter = 100000

        os.makedirs(self.output_dir, exist_ok=True)
        self.raw_csv_path = os.path.join(self.output_dir, "phase14_raw.csv")
        self.manifest_path = os.path.join(self.output_dir, "experiment_manifest.json")
        self.env_path = os.path.join(self.output_dir, "environment.json")

    def capture_environment(self) -> Dict[str, Any]:
        """Record system environment for scientific reproducibility."""
        env = {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "lb_port": self.lb_port,
            "backend_ports": self.server_ports,
            "algorithms": ALGORITHMS,
            "regimes": REGIMES,
            "repetitions": self.repetitions,
            "base_seeds": self.base_seeds,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.env_path, "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2)
        return env

    def capture_cluster_pre_snapshot(self) -> Dict[str, Any]:
        """Capture real-time pre-routing metrics snapshot across all 3 backends."""
        metrics_by_backend = self.cluster.collector.collect_all()
        snapshot: Dict[str, Any] = {}
        for i, port in enumerate(self.server_ports, start=1):
            url = f"http://127.0.0.1:{port}"
            m: Optional[ServerMetrics] = metrics_by_backend.get(url)
            snapshot[f"s{i}_cpu_pre"] = round(m.cpu_percent, 2) if m else 0.0
            snapshot[f"s{i}_mem_pre"] = round(m.memory_percent, 2) if m else 0.0
            snapshot[f"s{i}_conn_pre"] = int(m.active_connections) if m else 0
            snapshot[f"s{i}_resp_pre"] = round(m.avg_response_time_ms, 2) if m else 0.0
            snapshot[f"s{i}_lat_pre"] = round(m.network_latency_ms, 2) if m else 0.0
        return snapshot

    def execute_request(
        self,
        experiment_id: str,
        run_idx: int,
        algorithm: str,
        regime: str,
        request_idx: int,
        req_duration: float,
        concurrency: int,
        request_rate: float,
        seed: int,
        req_type: str = "normal",
        priority_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a single HTTP request through the load balancer and record metrics."""
        self.global_request_counter += 1
        request_id = f"p14_{regime[:3]}_{request_idx:04d}_{self.global_request_counter}"
        endpoint = f"/process?duration={req_duration:.4f}"
        url = f"http://127.0.0.1:{self.lb_port}{endpoint}"

        # Capture pre-routing state
        pre_snapshot = self.capture_cluster_pre_snapshot()
        optimal_server = calculate_optimal_server(pre_snapshot)

        # Prepare HTTP request
        req = urllib.request.Request(
            url=url,
            headers={
                "X-Request-ID": request_id,
                "X-Client-Seed": str(seed),
                "User-Agent": "Phase14Benchmark/1.0",
            },
        )
        is_priority = priority_meta is not None
        priority_val = None
        deadline_ms = None
        if is_priority:
            priority_val = priority_meta["priority"]
            deadline_ms = priority_meta["deadline_ms"]
            req.add_header("X-Priority", priority_val)
            req.add_header("X-Deadline", str(time.time() + (deadline_ms / 1000.0)))
            req.add_header("X-Estimated-Duration", f"{req_duration:.4f}")

        t0 = time.perf_counter()
        resp_status = 500
        success = False
        backend_server = None
        resp_headers: Dict[str, str] = {}
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                resp_status = resp.status
                success = (200 <= resp_status < 400)
                resp.read()
                backend_server = resp.headers.get("X-Backend-Server")
                for k, v in resp.headers.items():
                    resp_headers[k.lower()] = v
        except urllib.error.HTTPError as e:
            resp_status = e.code
            success = False
        except Exception:
            resp_status = 504
            success = False
        t1 = time.perf_counter()
        response_time_ms = max(0.1, round((t1 - t0) * 1000.0, 3))

        # Extract routing overhead from headers
        overhead_hdr = resp_headers.get("x-routing-overhead-ms")
        if overhead_hdr:
            try:
                routing_overhead_ms = max(0.0, float(overhead_hdr))
            except Exception:
                routing_overhead_ms = 0.01
        else:
            routing_overhead_ms = 0.01

        # Extract ML / Adaptive observability
        ml_pred = resp_headers.get("x-ml-predicted-server", "")
        conf_hdr = resp_headers.get("x-ml-confidence")
        ml_conf = float(conf_hdr) if conf_hdr else None
        ml_fb = resp_headers.get("x-ml-fallback", "false").lower() == "true"
        selected_model = resp_headers.get("x-selected-model", "")
        sel_time_hdr = resp_headers.get("x-model-selection-time")
        sel_time_ms = float(sel_time_hdr) if sel_time_hdr else 0.0

        # Determine chosen server label
        port_to_label = {
            f"http://127.0.0.1:{self.server_ports[0]}": "server-1",
            f"http://127.0.0.1:{self.server_ports[1]}": "server-2",
            f"http://127.0.0.1:{self.server_ports[2]}": "server-3",
        }
        chosen_label = port_to_label.get(backend_server, "server-1")
        suboptimal_routing = 0 if chosen_label == optimal_server else 1

        deadline_met = None
        if is_priority and deadline_ms is not None:
            deadline_met = 1 if response_time_ms <= deadline_ms else 0

        record = {
            "experiment_id": experiment_id,
            "run_idx": run_idx,
            "algorithm": algorithm,
            "regime": regime,
            "request_id": request_id,
            "timestamp": round(time.time(), 4),
            "request_type": req_type,
            "request_duration": round(req_duration, 4),
            "concurrency": concurrency,
            "request_rate": request_rate,
            "seed": seed,
            "backend_server": backend_server or f"http://127.0.0.1:{self.server_ports[0]}",
            "success": success,
            "status_code": resp_status,
            "response_time_ms": response_time_ms,
            "routing_overhead_ms": routing_overhead_ms,
            "ml_predicted_server": ml_pred,
            "ml_confidence": ml_conf,
            "ml_fallback": ml_fb,
            "selected_model": selected_model,
            "model_selection_time_ms": sel_time_ms,
            "s1_cpu_pre": pre_snapshot["s1_cpu_pre"],
            "s1_mem_pre": pre_snapshot["s1_mem_pre"],
            "s1_conn_pre": pre_snapshot["s1_conn_pre"],
            "s1_resp_pre": pre_snapshot["s1_resp_pre"],
            "s1_lat_pre": pre_snapshot["s1_lat_pre"],
            "s2_cpu_pre": pre_snapshot["s2_cpu_pre"],
            "s2_mem_pre": pre_snapshot["s2_mem_pre"],
            "s2_conn_pre": pre_snapshot["s2_conn_pre"],
            "s2_resp_pre": pre_snapshot["s2_resp_pre"],
            "s2_lat_pre": pre_snapshot["s2_lat_pre"],
            "s3_cpu_pre": pre_snapshot["s3_cpu_pre"],
            "s3_mem_pre": pre_snapshot["s3_mem_pre"],
            "s3_conn_pre": pre_snapshot["s3_conn_pre"],
            "s3_resp_pre": pre_snapshot["s3_resp_pre"],
            "s3_lat_pre": pre_snapshot["s3_lat_pre"],
            "optimal_server": optimal_server,
            "suboptimal_routing": suboptimal_routing,
            "is_priority_subset": is_priority,
            "priority": priority_val if is_priority else "",
            "deadline_ms": deadline_ms if is_priority else "",
            "deadline_met": deadline_met if is_priority else "",
        }
        return record

    def run_regime_workload(
        self,
        algorithm: str,
        regime: str,
        run_idx: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """Generate and execute matched workload for a specific regime and algorithm."""
        experiment_id = f"exp_p14_{regime}_{algorithm}_r{run_idx}"
        rng = random.Random(seed)

        # Regimes parameter profiles
        if regime == "low_load":
            num_req = 12
            concurrency = 2
            req_rate = 10.0
            durations = [0.005] * num_req
            req_types = ["normal"] * num_req
        elif regime == "medium_load":
            num_req = 22
            concurrency = 4
            req_rate = 20.0
            durations = [0.010] * num_req
            req_types = ["normal"] * num_req
        elif regime == "high_load":
            num_req = 32
            concurrency = 8
            req_rate = 35.0
            durations = [0.018] * num_req
            req_types = ["normal"] * num_req
        elif regime == "burst_load":
            num_req = 36
            concurrency = 12
            req_rate = 50.0
            durations = [rng.choice([0.005, 0.020]) for _ in range(num_req)]
            req_types = ["burst"] * num_req
        elif regime == "cpu_heavy":
            num_req = 24
            concurrency = 6
            req_rate = 15.0
            durations = [0.035] * num_req
            req_types = ["cpu_heavy"] * num_req
        elif regime == "mixed_workload":
            num_req = 26
            concurrency = 5
            req_rate = 20.0
            durations = [rng.choice([0.005, 0.015, 0.030]) for _ in range(num_req)]
            req_types = [rng.choice(["fast", "medium", "slow"]) for _ in range(num_req)]
        elif regime == "dynamic_workload":
            num_req = 28
            concurrency = 7
            req_rate = 25.0
            durations = [rng.uniform(0.005, 0.025) for _ in range(num_req)]
            req_types = ["dynamic"] * num_req
        elif regime == "queue_contention":
            num_req = 30
            concurrency = 10
            req_rate = 40.0
            durations = [0.025] * num_req
            req_types = ["contention"] * num_req
        elif regime == "backend_imbalance":
            num_req = 25
            concurrency = 5
            req_rate = 20.0
            durations = [0.012] * num_req
            req_types = ["imbalance"] * num_req
        else:
            num_req = 20
            concurrency = 4
            req_rate = 15.0
            durations = [0.010] * num_req
            req_types = ["default"] * num_req

        # For backend_imbalance, start real CPU stress against backend 1
        stress_stop = None
        stress_thread = None
        if regime == "backend_imbalance":
            stress_stop = threading.Event()
            stress_thread = start_backend_stress(
                backend_url=f"http://127.0.0.1:{self.server_ports[0]}",
                duration=0.08,
                stop_event=stress_stop,
            )
            time.sleep(0.08)  # Let server 1 accumulate load

        records = []
        try:
            # Execute with thread pool honoring concurrency
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                interval = 1.0 / req_rate if req_rate > 0 else 0.0
                for req_i in range(num_req):
                    f = executor.submit(
                        self.execute_request,
                        experiment_id=experiment_id,
                        run_idx=run_idx,
                        algorithm=algorithm,
                        regime=regime,
                        request_idx=req_i + 1,
                        req_duration=durations[req_i],
                        concurrency=concurrency,
                        request_rate=req_rate,
                        seed=seed,
                        req_type=req_types[req_i],
                    )
                    futures.append(f)
                    if interval > 0:
                        time.sleep(interval * 0.3)

                for f in as_completed(futures):
                    records.append(f.result())
        finally:
            if stress_stop:
                stress_stop.set()
                if stress_thread:
                    stress_thread.join(timeout=1.0)

        # Sort records by request_id
        records.sort(key=lambda r: r["request_id"])
        return records

    def run_priority_evaluation(self) -> List[Dict[str, Any]]:
        """Run isolated Priority & Deadline evaluation across 4 algorithms and 2 high-stress regimes."""
        logger.info("Executing isolated Priority and Deadline-Aware evaluation...")
        priority_algorithms = [
            ("round_robin", "priority_round_robin"),
            ("least_connections", "priority_least_connections"),
            ("logistic_regression", "priority_ml"),
            ("adaptive_policy", "priority_adaptive"),
        ]
        priority_levels = [
            ("LOW", 250.0, 0.010),
            ("NORMAL", 120.0, 0.015),
            ("HIGH", 70.0, 0.020),
            ("CRITICAL", 45.0, 0.025),
        ]
        stress_regimes = ["queue_contention", "backend_imbalance"]
        priority_records = []

        for p_base, p_name in priority_algorithms:
            self.cluster.set_router(p_base, priority=True)
            for regime in stress_regimes:
                for rep in range(1, 4):  # 3 matched repetitions
                    seed = self.base_seeds[rep - 1]
                    rng = random.Random(seed)
                    experiment_id = f"exp_p14_prio_{regime}_{p_name}_r{rep}"

                    stress_stop = None
                    if regime == "backend_imbalance":
                        stress_stop = threading.Event()
                        start_backend_stress(
                            backend_url=f"http://127.0.0.1:{self.server_ports[0]}",
                            duration=0.08,
                            stop_event=stress_stop,
                        )
                        time.sleep(0.08)

                    try:
                        with ThreadPoolExecutor(max_workers=6) as executor:
                            futures = []
                            for req_i in range(20):
                                prio_choice, dline, dur = rng.choice(priority_levels)
                                p_meta = {"priority": prio_choice, "deadline_ms": dline}
                                f = executor.submit(
                                    self.execute_request,
                                    experiment_id=experiment_id,
                                    run_idx=rep,
                                    algorithm=p_name,
                                    regime=regime,
                                    request_idx=req_i + 1,
                                    req_duration=dur,
                                    concurrency=6,
                                    request_rate=25.0,
                                    seed=seed,
                                    req_type="priority",
                                    priority_meta=p_meta,
                                )
                                futures.append(f)
                                time.sleep(0.01)

                            for f in as_completed(futures):
                                priority_records.append(f.result())
                    finally:
                        if stress_stop:
                            stress_stop.set()

        logger.info("Completed priority evaluation with %d records", len(priority_records))
        return priority_records

    def run_all(self) -> Tuple[str, str, str]:
        """Execute the complete Phase 14 benchmark suite."""
        logger.info("Starting Phase 14 End-to-End System Benchmark...")
        t_suite_start = time.time()
        self.capture_environment()
        self.cluster.start()

        all_records: List[Dict[str, Any]] = []
        manifest_runs: List[Dict[str, Any]] = []

        try:
            total_runs = len(ALGORITHMS) * len(REGIMES) * self.repetitions
            run_count = 0

            for alg in ALGORITHMS:
                logger.info("Configuring Load Balancer for algorithm: %s", alg)
                self.cluster.set_router(alg)
                time.sleep(0.05)

                for regime in REGIMES:
                    for rep in range(1, self.repetitions + 1):
                        run_count += 1
                        seed = self.base_seeds[rep - 1]
                        logger.info(
                            "[%d/%d] Running %s on %s (Rep %d, Seed %d)",
                            run_count,
                            total_runs,
                            alg,
                            regime,
                            rep,
                            seed,
                        )

                        records = self.run_regime_workload(
                            algorithm=alg,
                            regime=regime,
                            run_idx=rep,
                            seed=seed,
                        )
                        all_records.extend(records)

                        manifest_runs.append({
                            "experiment_id": f"exp_p14_{regime}_{alg}_r{rep}",
                            "algorithm": alg,
                            "regime": regime,
                            "run_idx": rep,
                            "seed": seed,
                            "request_count": len(records),
                        })

            # Execute Priority & Deadline extension
            priority_records = self.run_priority_evaluation()
            all_records.extend(priority_records)

        finally:
            self.cluster.stop()

        # Write data/phase14/phase14_raw.csv
        with open(self.raw_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_records)
        logger.info("Persisted %d raw observations to %s", len(all_records), self.raw_csv_path)

        # Write data/phase14/experiment_manifest.json
        manifest = {
            "phase": 14,
            "total_records": len(all_records),
            "core_benchmark_records": len(all_records) - len(priority_records),
            "priority_records": len(priority_records),
            "total_benchmark_runs": len(manifest_runs),
            "algorithms": ALGORITHMS,
            "regimes": REGIMES,
            "repetitions": self.repetitions,
            "base_seeds": self.base_seeds,
            "runs": manifest_runs,
            "duration_seconds": round(time.time() - t_suite_start, 2),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Persisted experiment manifest to %s", self.manifest_path)

        return self.raw_csv_path, self.manifest_path, self.env_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 14 End-to-End System Benchmark")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for datasets")
    parser.add_argument("--repetitions", type=int, default=5, help="Repetitions per regime x algorithm")
    parser.add_argument("--lb-port", type=int, default=8000, help="Load Balancer port")
    args = parser.parse_args()

    runner = Phase14BenchmarkRunner(
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        lb_port=args.lb_port,
    )
    runner.run_all()

