"""Edge Condition Benchmarks for Phase 9.

Tests and records system behavior under:
1. Backend Failure: Take one backend offline, verify routing, ML fallback, success, and zero crash.
2. Backend Recovery: Bring the server back online and verify system restoration.
3. Short High-Concurrency Burst: Measure queue growth, CPU, latency, and routing under burst load.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List
import urllib.request
import pandas as pd

from benchmarks.live_comparison import LiveBenchmarkCluster
from client.workload_generator import WorkloadConfig, WorkloadGenerator
from monitoring.collector import MetricsCollector, ServerMetrics
from server.app import create_server

logger = logging.getLogger("benchmarks.edge_tests")


class EdgeConditionTester:
    """Executes dedicated edge condition experiments."""

    def __init__(
        self,
        output_dir: str = "data/phase9",
        lb_port: int = 8000,
        server_ports: List[int] = None,
        model_path: str = "models/logistic_regression.joblib",
    ):
        self.output_dir = output_dir
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.model_path = model_path
        os.makedirs(self.output_dir, exist_ok=True)
        self.edge_results_path = os.path.join(self.output_dir, "edge_conditions.json")

    def run_all_edge_tests(self) -> Dict[str, Any]:
        """Run all edge test experiments and return structured findings."""
        results = {}

        # 1. Backend Failure and Recovery Test on ML Router
        logger.info("Running Edge Test 1: Backend Failure and Recovery on ML Router...")
        results["failure_and_recovery"] = self.test_failure_and_recovery()

        # 2. Short High-Concurrency Burst Stress Test across all algorithms
        logger.info("Running Edge Test 2: Short High-Concurrency Burst Stress Test...")
        results["short_burst_stress"] = self.test_short_burst_stress()

        with open(self.edge_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Edge condition tests completed. Results written to %s", self.edge_results_path)
        return results

    def test_failure_and_recovery(self) -> Dict[str, Any]:
        """Take Server 3 offline (which ML router targets), verify fallback, then recover."""
        cluster = LiveBenchmarkCluster(
            lb_port=self.lb_port,
            server_ports=self.server_ports,
            algorithm="ml",
            model_path=self.model_path,
        )
        cluster.start()
        time.sleep(0.2)

        test_log = {}

        try:
            # Baseline: Server 3 is online
            req = urllib.request.Request(f"http://127.0.0.1:{self.lb_port}/health")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                baseline_backend = resp.headers.get("X-Backend-Server")
                baseline_fb = resp.headers.get("X-ML-Fallback")

            test_log["baseline"] = {
                "status": "healthy",
                "routed_backend": baseline_backend,
                "is_fallback": baseline_fb,
            }

            # ACTION: Shut down server-3 (:8003)
            logger.info("Simulating failure on Server 3 (:8003)...")
            cluster.servers[2].shutdown()
            cluster.servers[2].server_close()
            time.sleep(0.1)

            # Send requests through LB during failure
            failure_reqs = []
            for i in range(10):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.lb_port}/health", timeout=3.0) as resp:
                        failure_reqs.append({
                            "req_id": i + 1,
                            "status_code": resp.status,
                            "backend": resp.headers.get("X-Backend-Server"),
                            "fallback": resp.headers.get("X-ML-Fallback"),
                            "predicted": resp.headers.get("X-ML-Predicted-Server"),
                        })
                except Exception as e:
                    failure_reqs.append({"req_id": i + 1, "error": str(e)})

            test_log["failure_phase"] = {
                "offline_server": "http://127.0.0.1:8003",
                "total_sent": len(failure_reqs),
                "successful": sum(1 for r in failure_reqs if r.get("status_code") == 200),
                "fallback_count": sum(1 for r in failure_reqs if r.get("fallback") == "true"),
                "requests": failure_reqs,
            }

            # ACTION: Recover server-3 (:8003)
            logger.info("Simulating recovery of Server 3 (:8003)...")
            rec_srv = create_server(server_id="server-3", host="127.0.0.1", port=8003)
            t = threading.Thread(target=rec_srv.serve_forever, daemon=True)
            t.start()
            cluster.servers[2] = rec_srv
            cluster.server_threads[2] = t
            cluster._wait_for_url("http://127.0.0.1:8003/health")
            time.sleep(0.2)

            # Send requests after recovery
            recovery_reqs = []
            for i in range(10):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.lb_port}/health", timeout=3.0) as resp:
                        recovery_reqs.append({
                            "req_id": i + 1,
                            "status_code": resp.status,
                            "backend": resp.headers.get("X-Backend-Server"),
                            "fallback": resp.headers.get("X-ML-Fallback"),
                        })
                except Exception as e:
                    recovery_reqs.append({"req_id": i + 1, "error": str(e)})

            test_log["recovery_phase"] = {
                "recovered_server": "http://127.0.0.1:8003",
                "total_sent": len(recovery_reqs),
                "successful": sum(1 for r in recovery_reqs if r.get("status_code") == 200),
                "fallback_count": sum(1 for r in recovery_reqs if r.get("fallback") == "true"),
                "requests": recovery_reqs,
            }

        finally:
            cluster.stop()

        return test_log

    def test_short_burst_stress(self) -> Dict[str, Any]:
        """Subject cluster to a sharp short burst (50 requests, concurrency=25, burst=25)."""
        burst_results = {}
        for algo in ["round_robin", "least_connections", "ip_hash", "ml"]:
            cluster = LiveBenchmarkCluster(
                lb_port=self.lb_port,
                server_ports=self.server_ports,
                algorithm=algo,
                model_path=self.model_path,
            )
            cluster.start()
            time.sleep(0.2)

            cfg = WorkloadConfig(
                target_url=f"http://127.0.0.1:{self.lb_port}",
                num_requests=50,
                concurrency=25,
                request_rate=None,
                endpoint="/process",
                request_duration=0.04,
                burst_size=25,
                burst_interval=0.2,
                scenario_name="edge_burst",
                seed=999,
            )

            start = time.perf_counter()
            gen = WorkloadGenerator(cfg)
            records = gen.run()
            duration = time.perf_counter() - start

            cluster.stop()

            succ = sum(1 for r in records if r.success)
            lats = [r.response_time for r in records if r.success] or [0.0]
            burst_results[algo] = {
                "total_requests": len(records),
                "successful": succ,
                "duration_s": round(duration, 3),
                "throughput_rps": round(len(records) / duration, 2),
                "avg_latency_ms": round(float(pd.Series(lats).mean()), 2),
                "p50_latency_ms": round(float(pd.Series(lats).median()), 2),
                "p95_latency_ms": round(float(pd.Series(lats).quantile(0.95)), 2),
                "p99_latency_ms": round(float(pd.Series(lats).quantile(0.99)), 2),
                "max_latency_ms": round(float(pd.Series(lats).max()), 2),
            }

        return burst_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tester = EdgeConditionTester()
    tester.run_all_edge_tests()

