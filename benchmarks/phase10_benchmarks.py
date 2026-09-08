"""Phase 10 Priority & Deadline Aware Benchmark Suite.

Executes controlled, reproducible performance comparisons between:
1. Standard ML Routing (Logistic Regression)
2. Priority & Deadline Aware ML Routing (PriorityDeadlineRouter)

Under identical controlled scenarios:
- Scenario A: Equal Priority
- Scenario B: Mixed Priority
- Scenario C: Tight Deadlines
- Scenario D: Priority + Deadline Conflict
- Scenario E: Backend Stress
- Scenario F: Backend Failure
- Scenario G: Queue Contention
"""

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

from benchmarks.live_comparison import LiveBenchmarkCluster
from client.workload_generator import WorkloadConfig, WorkloadGenerator
from config.backends import DEFAULT_BACKENDS
from load_balancer.app import create_load_balancer
from monitoring.collector import MetricsCollector, ServerMetrics
from server.app import create_server

logger = logging.getLogger("benchmarks.phase10_benchmarks")

PHASE10_SCENARIOS = [
    "scenario_a_equal_priority",
    "scenario_b_mixed_priority",
    "scenario_c_tight_deadlines",
    "scenario_d_priority_deadline_conflict",
    "scenario_e_backend_stress",
    "scenario_f_backend_failure",
    "scenario_g_queue_contention",
]

PHASE10_ALGORITHMS = ["ml", "priority_ml"]


class Phase10BenchmarkRunner:
    """Orchestrates Phase 10 priority & deadline benchmarks."""

    def __init__(
        self,
        output_dir: str = "data/phase10",
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
        """Record reproducibility metadata for Phase 10."""
        meta = {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "backend_ports": self.server_ports,
            "load_balancer_port": self.lb_port,
            "algorithms": PHASE10_ALGORITHMS,
            "scenarios": PHASE10_SCENARIOS,
            "repetitions": self.repetitions,
            "base_seed": self.base_seed,
            "model_path": self.model_path,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(self.environment_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

    def _build_scenario_config(self, scenario_name: str, seed: int, experiment_id: str) -> WorkloadConfig:
        """Create WorkloadConfig tailored to each specific Phase 10 scenario."""
        target_url = f"http://127.0.0.1:{self.lb_port}"

        if scenario_name == "scenario_a_equal_priority":
            return WorkloadConfig(
                target_url=target_url,
                num_requests=30,
                concurrency=5,
                request_rate=10.0,
                endpoint="/process",
                request_duration=0.03,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="equal",
            )
        elif scenario_name == "scenario_b_mixed_priority":
            return WorkloadConfig(
                target_url=target_url,
                num_requests=40,
                concurrency=6,
                request_rate=15.0,
                endpoint="/process",
                request_duration=0.03,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="mixed",
            )
        elif scenario_name == "scenario_c_tight_deadlines":
            return WorkloadConfig(
                target_url=target_url,
                num_requests=40,
                concurrency=8,
                request_rate=None,  # unthrottled concurrent burst
                endpoint="/process",
                request_duration=0.03,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="tight_deadlines",
            )
        elif scenario_name == "scenario_d_priority_deadline_conflict":
            return WorkloadConfig(
                target_url=target_url,
                num_requests=30,
                concurrency=6,
                request_rate=None,
                endpoint="/process",
                request_duration=0.04,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="conflict",
            )
        elif scenario_name == "scenario_e_backend_stress":
            # Sustained unthrottled requests with mixed priorities under heavy computation
            return WorkloadConfig(
                target_url=target_url,
                num_requests=50,
                concurrency=10,
                request_rate=None,
                endpoint="/process",
                request_duration=0.06,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="mixed",
            )
        elif scenario_name == "scenario_f_backend_failure":
            return WorkloadConfig(
                target_url=target_url,
                num_requests=30,
                concurrency=5,
                request_rate=10.0,
                endpoint="/process",
                request_duration=0.03,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="mixed",
            )
        elif scenario_name == "scenario_g_queue_contention":
            # High concurrency queue contention (20 concurrent workers)
            return WorkloadConfig(
                target_url=target_url,
                num_requests=60,
                concurrency=20,
                request_rate=None,
                endpoint="/process",
                request_duration=0.05,
                seed=seed,
                scenario_name=scenario_name,
                experiment_id=experiment_id,
                priority_profile="mixed",
            )
        else:
            raise ValueError(f"Unknown Phase 10 scenario: {scenario_name}")

    def run_single_benchmark(
        self,
        algorithm: str,
        scenario: str,
        run_idx: int,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run an individual experiment for algorithm x scenario x run_idx."""
        experiment_id = f"p10_{scenario}_{algorithm}_r{run_idx}"

        # Spin up cluster
        b_timeout = 0.3 if scenario == "scenario_f_backend_failure" else 2.0
        cluster = LiveBenchmarkCluster(
            lb_port=self.lb_port,
            server_ports=self.server_ports,
            algorithm=algorithm,
            model_path=self.model_path,
            backend_timeout=b_timeout,
        )
        cluster.start()
        time.sleep(0.2)

        # For Scenario F (backend failure), deliberately kill Server 3 before workload
        if scenario == "scenario_f_backend_failure":
            logger.info("Scenario F: Simulating Server 3 failure before dispatch...")
            cluster.servers[2].shutdown()
            cluster.servers[2].server_close()
            time.sleep(0.1)

        cfg = self._build_scenario_config(scenario, seed=seed, experiment_id=experiment_id)

        bench_start = time.perf_counter()
        gen = WorkloadGenerator(cfg)
        records = gen.run()
        bench_duration = time.perf_counter() - bench_start

        cluster.stop()

        # Parse request records and priority headers
        raw_rows = []
        backend_counts = {f"http://127.0.0.1:{p}": 0 for p in self.server_ports}
        deadlines_met = 0
        deadlines_violated = 0
        lateness_values = []
        priority_latencies = {"LOW": [], "NORMAL": [], "HIGH": [], "CRITICAL": []}
        priority_violations = {"LOW": 0, "NORMAL": 0, "HIGH": 0, "CRITICAL": 0}
        priority_counts = {"LOW": 0, "NORMAL": 0, "HIGH": 0, "CRITICAL": 0}
        override_count = 0

        for r in records:
            backend_counts[r.backend_server] = backend_counts.get(r.backend_server, 0) + 1
            headers = r.response_headers or {}

            p_val = headers.get("X-Request-Priority") or "NORMAL"
            p_override = headers.get("X-Priority-Override") == "true"
            d_override = headers.get("X-Deadline-Override") == "true"
            r_reason = headers.get("X-Routing-Reason") or "normal"
            ml_pred = headers.get("X-ML-Predicted-Server") or "unknown"
            final_b = headers.get("X-Final-Backend") or r.backend_server or "unknown"

            slack_str = headers.get("X-Deadline-Slack")
            slack_ms = float(slack_str) if slack_str is not None else None

            # Deadline evaluation
            is_deadline_met = True
            lateness_ms = 0.0
            if slack_ms is not None:
                # If duration of execution exceeded slack available at arrival
                if r.response_time > slack_ms:
                    is_deadline_met = False
                    lateness_ms = round(r.response_time - slack_ms, 2)
                    deadlines_violated += 1
                    priority_violations[p_val] = priority_violations.get(p_val, 0) + 1
                else:
                    deadlines_met += 1
                lateness_values.append(lateness_ms)

            if p_override or d_override:
                override_count += 1

            priority_counts[p_val] = priority_counts.get(p_val, 0) + 1
            if r.success:
                priority_latencies[p_val].append(r.response_time)

            raw_rows.append({
                "experiment_id": experiment_id,
                "run_id": run_idx,
                "algorithm": algorithm,
                "scenario": scenario,
                "request_id": r.request_id,
                "timestamp": r.timestamp,
                "request_type": r.request_type,
                "request_duration": r.request_duration,
                "concurrency": cfg.concurrency,
                "backend_server": r.backend_server or "unknown",
                "success": r.success,
                "status_code": r.status_code,
                "response_time_ms": r.response_time,
                "priority": p_val,
                "deadline_slack_ms": slack_ms,
                "deadline_met": is_deadline_met,
                "lateness_ms": lateness_ms,
                "ml_predicted_server": ml_pred,
                "final_backend": final_b,
                "priority_override": p_override,
                "deadline_override": d_override,
                "routing_reason": r_reason,
            })

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

        total_deadline_reqs = deadlines_met + deadlines_violated
        violation_rate = (deadlines_violated / total_deadline_reqs * 100.0) if total_deadline_reqs > 0 else 0.0
        avg_lateness = float(np.mean(lateness_values)) if lateness_values else 0.0

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
            "deadlines_total": total_deadline_reqs,
            "deadlines_met": deadlines_met,
            "deadlines_violated": deadlines_violated,
            "deadline_violation_rate": round(violation_rate, 2),
            "avg_lateness_ms": round(avg_lateness, 2),
            "priority_overrides": override_count,
            "high_critical_avg_latency_ms": round(
                float(np.mean(priority_latencies["HIGH"] + priority_latencies["CRITICAL"])), 2
            ) if (priority_latencies["HIGH"] + priority_latencies["CRITICAL"]) else None,
            "low_normal_avg_latency_ms": round(
                float(np.mean(priority_latencies["LOW"] + priority_latencies["NORMAL"])), 2
            ) if (priority_latencies["LOW"] + priority_latencies["NORMAL"]) else None,
            "s1_requests": backend_counts.get(f"http://127.0.0.1:{self.server_ports[0]}", 0),
            "s2_requests": backend_counts.get(f"http://127.0.0.1:{self.server_ports[1]}", 0),
            "s3_requests": backend_counts.get(f"http://127.0.0.1:{self.server_ports[2]}", 0),
        }

        return raw_rows, summary

    def run_full_suite(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute the full experimental suite across all scenarios, algorithms, and repetitions."""
        self.capture_environment_metadata()
        all_raw = []
        all_summaries = []

        total_runs = len(PHASE10_SCENARIOS) * len(PHASE10_ALGORITHMS) * self.repetitions
        completed = 0
        suite_start = time.time()

        logger.info(
            "Starting Phase 10 Priority Benchmark: %d total runs (%d scenarios x %d algos x %d reps)...",
            total_runs,
            len(PHASE10_SCENARIOS),
            len(PHASE10_ALGORITHMS),
            self.repetitions,
        )

        for scenario in PHASE10_SCENARIOS:
            for rep in range(1, self.repetitions + 1):
                seed = self.base_seed + rep - 1
                for algo in PHASE10_ALGORITHMS:
                    completed += 1
                    logger.info(
                        "[%d/%d] Phase 10 test: scenario=%s algo=%s rep=%d...",
                        completed,
                        total_runs,
                        scenario,
                        algo,
                        rep,
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
            "Phase 10 benchmarks finished in %.2fs. Saved %d raw records to %s and summaries to %s",
            elapsed,
            len(df_raw),
            self.raw_results_path,
            self.run_summaries_path,
        )

        return df_raw, df_summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 10 Priority & Deadline Benchmark Suite")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per scenario x algorithm")
    parser.add_argument("--output-dir", default="data/phase10", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    runner = Phase10BenchmarkRunner(
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        base_seed=args.seed,
    )
    runner.run_full_suite()

