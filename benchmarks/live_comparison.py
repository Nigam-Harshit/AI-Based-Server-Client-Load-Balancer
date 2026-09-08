"""Phase 9 Live Load Balancing Benchmark Suite.

Executes rigorous, controlled, and reproducible performance comparisons
between traditional routing algorithms (Round Robin, Least Connections, IP Hash)
and Machine Learning (Logistic Regression).
"""

import csv
import json
import logging
import os
import platform
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import numpy as np
import pandas as pd
import psutil

from client.workload_generator import WorkloadConfig, WorkloadGenerator, get_scenario
from config.backends import DEFAULT_BACKENDS
from load_balancer.app import create_load_balancer
from monitoring.collector import MetricsCollector, ServerMetrics
from server.app import create_server

logger = logging.getLogger("benchmarks.live_comparison")

ALGORITHMS = ["round_robin", "least_connections", "ip_hash", "ml"]

# Exact scenarios mandated for Phase 9
SCENARIOS = [
    "low_traffic",
    "medium_traffic",
    "high_traffic",
    "burst_traffic",
    "cpu_heavy",
    "mixed",
    "dynamic",
]

RAW_FIELDS = [
    "experiment_id",
    "run_id",
    "algorithm",
    "scenario",
    "request_id",
    "timestamp",
    "request_type",
    "request_duration",
    "concurrency",
    "request_rate",
    "random_seed",
    "backend_server",
    "success",
    "status_code",
    "response_time_ms",
    "ml_predicted_server",
    "ml_confidence",
    "ml_fallback",
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
    "best_server",
]


class LiveBenchmarkCluster:
    """Manages the lifecycle of 3 HTTP backend servers and 1 central Load Balancer."""

    def __init__(
        self,
        lb_port: int = 8000,
        server_ports: Optional[List[int]] = None,
        algorithm: str = "round_robin",
        model_path: str = "models/logistic_regression.joblib",
        backend_timeout: float = 2.0,
    ):
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.backends = [f"http://127.0.0.1:{p}" for p in self.server_ports]
        self.algorithm = algorithm
        self.model_path = model_path
        self.backend_timeout = backend_timeout
        self.servers = []
        self.server_threads = []
        self.lb = None
        self.lb_thread = None

    def start(self) -> None:
        """Start backend servers and load balancer, waiting until ready."""
        # 1. Start 3 backend server instances
        for i, port in enumerate(self.server_ports, start=1):
            srv = create_server(server_id=f"server-{i}", host="127.0.0.1", port=port)
            t = threading.Thread(target=srv.serve_forever, daemon=True)
            t.start()
            self.servers.append(srv)
            self.server_threads.append(t)

        for port in self.server_ports:
            self._wait_for_url(f"http://127.0.0.1:{port}/health")

        # 2. Start Load Balancer
        self.lb = create_load_balancer(
            host="127.0.0.1",
            port=self.lb_port,
            algorithm=self.algorithm,
            backends=self.backends,
            model_path=self.model_path,
            backend_timeout=self.backend_timeout,
        )
        self.lb_thread = threading.Thread(target=self.lb.serve_forever, daemon=True)
        self.lb_thread.start()
        self._wait_for_url(f"http://127.0.0.1:{self.lb_port}/health")

    def stop(self) -> None:
        """Cleanly stop all load balancer and server instances."""
        if self.lb:
            try:
                self.lb.shutdown()
                self.lb.server_close()
            except Exception:
                pass
        for s in self.servers:
            try:
                s.shutdown()
                s.server_close()
            except Exception:
                pass

    def _wait_for_url(self, url: str, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise TimeoutError(f"Service at {url} failed to respond within {timeout}s")


class Phase9BenchmarkRunner:
    """Executes the Phase 9 experimental evaluation matrix."""

    def __init__(
        self,
        output_dir: str = "data/phase9",
        repetitions: int = 3,
        base_seed: int = 42,
        model_path: str = "models/logistic_regression.joblib",
        lb_port: int = 8000,
        server_ports: Optional[List[int]] = None,
    ):
        self.output_dir = output_dir
        self.repetitions = repetitions
        self.base_seed = base_seed
        self.model_path = model_path
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.backends = [f"http://127.0.0.1:{p}" for p in self.server_ports]
        self.collector = MetricsCollector(backends=self.backends)

        os.makedirs(self.output_dir, exist_ok=True)
        self.raw_results_path = os.path.join(self.output_dir, "raw_results.csv")
        self.run_summaries_path = os.path.join(self.output_dir, "run_summaries.csv")
        self.environment_path = os.path.join(self.output_dir, "environment.json")

    def capture_environment_metadata(self) -> Dict[str, Any]:
        """Record precise execution environment specifications for scientific reproducibility."""
        meta = {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "backend_ports": self.server_ports,
            "backend_count": len(self.server_ports),
            "load_balancer_port": self.lb_port,
            "algorithms": ALGORITHMS,
            "scenarios": SCENARIOS,
            "repetitions": self.repetitions,
            "base_seed": self.base_seed,
            "model_path": self.model_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.environment_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    def compute_optimal_backend(self, snapshot: Dict[str, Any]) -> Optional[str]:
        """Compute optimal backend label independently using Phase 5 formula."""
        costs = []
        for i in range(1, 4):
            net = float(snapshot.get(f"s{i}_lat_pre", 0.0))
            resp = float(snapshot.get(f"s{i}_resp_pre", 0.0))
            conn = int(snapshot.get(f"s{i}_conn_pre", 0))
            cost = net + resp * (1.0 + conn)
            costs.append((f"server-{i}", cost))

        c_vals = [c for _, c in costs]
        if max(c_vals) - min(c_vals) <= 2.0:
            return None
        return min(costs, key=lambda x: x[1])[0]

    def run_single_benchmark(
        self,
        algorithm: str,
        scenario: str,
        run_idx: int,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run a single algorithm x scenario x repetition test."""
        experiment_id = f"p9_{scenario}_{algorithm}_r{run_idx}"
        target_url = f"http://127.0.0.1:{self.lb_port}"

        # 1. Spin up cluster with the given algorithm
        cluster = LiveBenchmarkCluster(
            lb_port=self.lb_port,
            server_ports=self.server_ports,
            algorithm=algorithm,
            model_path=self.model_path,
        )
        cluster.start()

        time.sleep(0.2)  # Stabilization grace period

        # 2. Capture baseline cluster metrics before workload begins
        pre_metrics = self.collector.collect_all()
        pre_snapshot = {}
        for i, port in enumerate(self.server_ports, start=1):
            b_url = f"http://127.0.0.1:{port}"
            m: ServerMetrics = pre_metrics.get(b_url)
            pre_snapshot[f"s{i}_cpu_pre"] = m.cpu_percent if m else 0.0
            pre_snapshot[f"s{i}_mem_pre"] = m.memory_percent if m else 0.0
            pre_snapshot[f"s{i}_conn_pre"] = m.active_connections if m else 0
            pre_snapshot[f"s{i}_resp_pre"] = m.avg_response_time_ms if m else 0.0
            pre_snapshot[f"s{i}_lat_pre"] = m.network_latency_ms if m else 0.0

        optimal_label = self.compute_optimal_backend(pre_snapshot)

        # 3. Build and execute workload
        cfg: WorkloadConfig = get_scenario(
            scenario,
            target_url=target_url,
            seed=seed,
            experiment_id=experiment_id,
        )

        during_samples: List[Dict[str, float]] = []
        stop_sampling = threading.Event()

        def sample_metrics_during():
            while not stop_sampling.is_set():
                try:
                    now_m = self.collector.collect_all()
                    sample = {
                        f"s{i}_cpu": m.cpu_percent for i, m in enumerate(now_m.values(), 1)
                    }
                    during_samples.append(sample)
                except Exception:
                    pass
                time.sleep(0.05)

        sampler_thread = threading.Thread(target=sample_metrics_during, daemon=True)
        sampler_thread.start()

        bench_start = time.perf_counter()
        gen = WorkloadGenerator(cfg)
        records = gen.run()
        bench_duration = time.perf_counter() - bench_start

        stop_sampling.set()
        sampler_thread.join(timeout=1.0)

        # 4. Capture post-workload metrics
        post_metrics = self.collector.collect_all()

        # Extract per-request records with pre-metrics
        raw_rows: List[Dict[str, Any]] = []
        ml_pred_count = 0
        ml_fallback_count = 0
        confidences = []
        backend_counts = {f"http://127.0.0.1:{p}": 0 for p in self.server_ports}

        # Query LB last prediction state if ML
        for r in records:
            backend_counts[r.backend_server] = backend_counts.get(r.backend_server, 0) + 1

            headers = r.response_headers or {}
            pred_srv = headers.get("X-ML-Predicted-Server") or headers.get("x-ml-predicted-server")
            conf_str = headers.get("X-ML-Confidence") or headers.get("x-ml-confidence")
            fallback_str = headers.get("X-ML-Fallback") or headers.get("x-ml-fallback")

            conf_val = float(conf_str) if conf_str is not None else None
            is_fb = (fallback_str.lower() == "true") if fallback_str else False

            if algorithm == "ml":
                ml_pred_count += 1
                if is_fb:
                    ml_fallback_count += 1
                if conf_val is not None:
                    confidences.append(conf_val)

            row = {
                "experiment_id": experiment_id,
                "run_id": run_idx,
                "algorithm": algorithm,
                "scenario": scenario,
                "request_id": r.request_id,
                "timestamp": r.timestamp,
                "request_type": r.request_type,
                "request_duration": r.request_duration,
                "concurrency": cfg.concurrency,
                "request_rate": cfg.request_rate if cfg.request_rate is not None else -1,
                "random_seed": seed,
                "backend_server": r.backend_server or "unknown",
                "success": r.success,
                "status_code": r.status_code,
                "response_time_ms": r.response_time,
                "ml_predicted_server": pred_srv,
                "ml_confidence": conf_val,
                "ml_fallback": is_fb,
                "best_server": optimal_label,
            }
            row.update(pre_snapshot)
            raw_rows.append(row)

        # Stop cluster cleanly
        cluster.stop()


        # Compute summary metrics
        total_reqs = len(records)
        succ_reqs = sum(1 for r in records if r.success)
        failed_reqs = total_reqs - succ_reqs
        succ_rate = (succ_reqs / total_reqs * 100.0) if total_reqs > 0 else 0.0
        throughput = total_reqs / bench_duration if bench_duration > 0 else 0.0

        latencies = [r.response_time for r in records if r.success] or [0.0]
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        avg_lat = float(np.mean(latencies))
        max_lat = float(np.max(latencies))

        # Average CPU during workload
        avg_cpus = []
        if during_samples:
            for s in during_samples:
                avg_cpus.append(np.mean(list(s.values())))
            workload_cpu = float(np.mean(avg_cpus))
        else:
            workload_cpu = float(np.mean([m.cpu_percent for m in post_metrics.values()]))

        summary = {
            "experiment_id": experiment_id,
            "run_id": run_idx,
            "algorithm": algorithm,
            "scenario": scenario,
            "random_seed": seed,
            "total_requests": total_reqs,
            "successful_requests": succ_reqs,
            "failed_requests": failed_reqs,
            "success_rate": round(succ_rate, 2),
            "throughput_rps": round(throughput, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "max_latency_ms": round(max_lat, 2),
            "workload_cpu_percent": round(workload_cpu, 2),
            "s1_requests": backend_counts.get(f"http://127.0.0.1:{self.server_ports[0]}", 0),
            "s2_requests": backend_counts.get(f"http://127.0.0.1:{self.server_ports[1]}", 0),
            "s3_requests": backend_counts.get(f"http://127.0.0.1:{self.server_ports[2]}", 0),
            "ml_prediction_count": ml_pred_count,
            "ml_fallback_count": ml_fallback_count,
            "ml_fallback_rate": (ml_fallback_count / ml_pred_count * 100.0) if ml_pred_count > 0 else 0.0,
            "ml_avg_confidence": round(float(np.mean(confidences)), 4) if confidences else None,
        }

        return raw_rows, summary

    def run_full_suite(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full experimental matrix across all algorithms, scenarios, and repetitions."""
        self.capture_environment_metadata()
        all_raw: List[Dict[str, Any]] = []
        all_summaries: List[Dict[str, Any]] = []

        total_runs = len(SCENARIOS) * len(ALGORITHMS) * self.repetitions
        completed = 0
        suite_start = time.time()

        logger.info(
            "Starting Phase 9 Live Benchmark: %d total runs (%d scenarios x %d algorithms x %d reps)...",
            total_runs,
            len(SCENARIOS),
            len(ALGORITHMS),
            self.repetitions,
        )

        for scenario in SCENARIOS:
            for rep in range(1, self.repetitions + 1):
                seed = self.base_seed + rep - 1
                for algo in ALGORITHMS:
                    completed += 1
                    logger.info(
                        "[%d/%d] Testing scenario=%s algo=%s rep=%d (seed=%d)...",
                        completed,
                        total_runs,
                        scenario,
                        algo,
                        rep,
                        seed,
                    )
                    raw_rows, summary = self.run_single_benchmark(
                        algorithm=algo,
                        scenario=scenario,
                        run_idx=rep,
                        seed=seed,
                    )
                    all_raw.extend(raw_rows)
                    all_summaries.append(summary)

        df_raw = pd.DataFrame(all_raw)
        df_summaries = pd.DataFrame(all_summaries)

        df_raw.to_csv(self.raw_results_path, index=False)
        df_summaries.to_csv(self.run_summaries_path, index=False)

        elapsed = round(time.time() - suite_start, 2)
        logger.info(
            "Completed Phase 9 Live Benchmark suite in %.2fs. Saved %d raw records to %s and summaries to %s",
            elapsed,
            len(df_raw),
            self.raw_results_path,
            self.run_summaries_path,
        )

        return df_raw, df_summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 9 Live Load Balancing Benchmark Suite")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per algorithm x scenario")
    parser.add_argument("--scenarios", nargs="*", default=None, help="Scenarios to run (default: all)")
    parser.add_argument("--algorithms", nargs="*", default=None, help="Algorithms to run (default: all)")
    parser.add_argument("--output-dir", default="data/phase9", help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runner = Phase9BenchmarkRunner(
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        base_seed=args.seed,
    )
    if args.scenarios:
        SCENARIOS = [s for s in args.scenarios if s in SCENARIOS]
    if args.algorithms:
        ALGORITHMS = [a for a in args.algorithms if a in ALGORITHMS]

    runner.run_full_suite()

