"""Phase 15: Deployment Readiness, Failure Injection & Sanity Evaluation Runner.

Executes:
1. Controlled Failure Injection Tests:
   - Test A: Stop server-1. Verify LB routes to healthy backends (server-2, server-3).
   - Test B: Stop server-2. Verify remaining healthy backends serve traffic.
   - Test C: Stop server-1 and server-2. Verify server-3 continues serving.
   - Test D: Stop all backends. Verify LB returns controlled HTTP error (502/503) without crashing.
   - Test E: Restart backend. Verify recovery and automatic re-inclusion.
2. Deployment Sanity Evaluation:
   - Baseline native workload execution vs degraded / recovered operations.
   - Metrics: success rate, P50 latency, P95 latency, throughput.
3. Persists results to:
   - data/phase15/failure_injection_results.json
   - data/phase15/deployment_sanity_results.json
   - experiments/results/phase15/01_failure_injection_timeline.png
"""

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from load_balancer.app import create_load_balancer
from server.app import create_server

logger = logging.getLogger("experiments.phase15_deployment_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "phase15")
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "phase15")
FAILURE_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "failure_injection_results.json")
SANITY_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "deployment_sanity_results.json")


def wait_for_url(url: str, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.05)
    return False


def make_request(url: str, timeout: float = 3.0) -> Tuple[int, Optional[str], float]:
    t0 = time.perf_counter()
    status = 500
    server_hdr = None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            status = resp.status
            server_hdr = resp.headers.get("X-Backend-Server")
            resp.read()
    except urllib.error.HTTPError as e:
        status = e.code
        server_hdr = e.headers.get("X-Backend-Server")
    except Exception:
        status = 504
    latency_ms = max(0.1, round((time.perf_counter() - t0) * 1000.0, 3))
    return status, server_hdr, latency_ms


class DeploymentFaultRunner:
    """Executes controlled deployment failure injection tests against live server instances."""

    def __init__(self, lb_port: int = 8000, server_ports: Optional[List[int]] = None):
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.backends = [f"http://127.0.0.1:{p}" for p in self.server_ports]
        self.servers: Dict[int, Any] = {}
        self.server_threads: Dict[int, threading.Thread] = {}
        self.lb = None
        self.lb_thread = None

    def start_backend(self, port: int, server_id: str) -> None:
        srv = create_server(server_id=server_id, host="127.0.0.1", port=port)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.servers[port] = srv
        self.server_threads[port] = t
        wait_for_url(f"http://127.0.0.1:{port}/health", timeout=3.0)

    def stop_backend(self, port: int) -> None:
        if port in self.servers:
            srv = self.servers.pop(port)
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
            time.sleep(0.1)

    def start_all(self) -> None:
        for i, port in enumerate(self.server_ports, start=1):
            self.start_backend(port, f"server-{i}")

        self.lb = create_load_balancer(
            host="127.0.0.1",
            port=self.lb_port,
            algorithm="least_connections",
            backends=self.backends,
            backend_timeout=1.5,
        )
        self.lb_thread = threading.Thread(target=self.lb.serve_forever, daemon=True)
        self.lb_thread.start()
        wait_for_url(f"http://127.0.0.1:{self.lb_port}/lb-health", timeout=3.0)

    def stop_all(self) -> None:
        if self.lb:
            try:
                self.lb.shutdown()
                self.lb.server_close()
            except Exception:
                pass
        for port in list(self.servers.keys()):
            self.stop_backend(port)

    def run_fault_tests(self) -> Dict[str, Any]:
        """Execute Tests A - E."""
        logger.info("Executing Phase 15 Failure Injection Tests...")
        self.start_all()
        results: Dict[str, Any] = {}
        timeline_events = []

        try:
            # Baseline sanity: all 3 healthy
            base_statuses = []
            for _ in range(6):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.005")
                base_statuses.append(st)
                timeline_events.append({"stage": "baseline", "status": st, "latency_ms": lat, "server": srv})
            results["baseline_3_backends"] = {
                "success_rate": round(base_statuses.count(200) / len(base_statuses) * 100.0, 1),
                "expected": 100.0,
                "passed": base_statuses.count(200) == len(base_statuses),
            }

            # Test A: Stop server-1
            logger.info("Test A: Stopping server-1 (:8001)...")
            self.stop_backend(self.server_ports[0])
            test_a_servers = set()
            test_a_statuses = []
            for _ in range(8):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.005")
                test_a_statuses.append(st)
                if srv:
                    test_a_servers.add(srv)
                timeline_events.append({"stage": "test_a_server1_down", "status": st, "latency_ms": lat, "server": srv})
            results["test_a_server_1_stopped"] = {
                "success_rate": round(test_a_statuses.count(200) / len(test_a_statuses) * 100.0, 1),
                "active_backends_observed": list(test_a_servers),
                "excluded_stopped_server_1": f"http://127.0.0.1:{self.server_ports[0]}" not in test_a_servers,
                "passed": f"http://127.0.0.1:{self.server_ports[0]}" not in test_a_servers and test_a_statuses.count(200) > 0,
            }

            # Test B: Stop server-2
            logger.info("Test B: Stopping server-2 (:8002)...")
            self.stop_backend(self.server_ports[1])
            test_b_statuses = []
            test_b_servers = set()
            for _ in range(6):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.005")
                test_b_statuses.append(st)
                if srv:
                    test_b_servers.add(srv)
                timeline_events.append({"stage": "test_b_server1_2_down", "status": st, "latency_ms": lat, "server": srv})
            results["test_b_server_2_stopped"] = {
                "success_rate": round(test_b_statuses.count(200) / len(test_b_statuses) * 100.0, 1),
                "active_backends_observed": list(test_b_servers),
                "sole_surviving_server_3": list(test_b_servers) == [f"http://127.0.0.1:{self.server_ports[2]}"],
                "passed": list(test_b_servers) == [f"http://127.0.0.1:{self.server_ports[2]}"],
            }

            # Test C: Verify single surviving backend (Server-3) maintains service
            logger.info("Test C: Validating single surviving backend (Server-3)...")
            results["test_c_single_survivor"] = {
                "surviving_backend": f"http://127.0.0.1:{self.server_ports[2]}",
                "traffic_sustained": test_b_statuses.count(200) > 0,
                "passed": test_b_statuses.count(200) > 0,
            }

            # Test D: Stop all backends (Server-3 stopped)
            logger.info("Test D: Stopping all backends (:8003 stopped)...")
            self.stop_backend(self.server_ports[2])
            test_d_statuses = []
            for _ in range(4):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.005")
                test_d_statuses.append(st)
                timeline_events.append({"stage": "test_d_all_down", "status": st, "latency_ms": lat, "server": srv})
            # Verify LB did not crash, responds with HTTP 502/503/504
            all_error_controlled = all(s in (502, 503, 504) for s in test_d_statuses)
            # Verify LB is still responsive on /lb-health
            lb_health_st, _, _ = make_request(f"http://127.0.0.1:{self.lb_port}/lb-health")
            results["test_d_all_backends_stopped"] = {
                "lb_survived_without_crashing": lb_health_st == 200,
                "controlled_error_codes": test_d_statuses,
                "passed": lb_health_st == 200 and all_error_controlled,
            }

            # Test E: Restart backend (Server-1 recovered)
            logger.info("Test E: Restarting backend server-1 (:8001)...")
            self.start_backend(self.server_ports[0], "server-1")
            time.sleep(0.2)
            test_e_statuses = []
            test_e_servers = set()
            for _ in range(6):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.005")
                test_e_statuses.append(st)
                if srv:
                    test_e_servers.add(srv)
                timeline_events.append({"stage": "test_e_server1_recovered", "status": st, "latency_ms": lat, "server": srv})
            results["test_e_backend_recovery"] = {
                "recovered_server": f"http://127.0.0.1:{self.server_ports[0]}",
                "traffic_resumed": test_e_statuses.count(200) > 0,
                "active_backends_observed": list(test_e_servers),
                "passed": test_e_statuses.count(200) > 0 and f"http://127.0.0.1:{self.server_ports[0]}" in test_e_servers,
            }

        finally:
            self.stop_all()

        # Save failure injection results
        with open(FAILURE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved failure injection results to %s", FAILURE_JSON_PATH)

        # Generate Failure Injection Timeline plot
        self._plot_timeline(timeline_events)
        return results

    def _plot_timeline(self, events: List[Dict[str, Any]]) -> None:
        """Plot response time and status code across the failure injection stages."""
        plt.figure(figsize=(10, 5))
        latencies = [e["latency_ms"] for e in events]
        statuses = [e["status"] for e in events]
        stages = [e["stage"] for e in events]
        x = range(len(events))

        colors = ["#59a14f" if s == 200 else "#e15759" for s in statuses]
        plt.scatter(x, latencies, c=colors, s=70, edgecolor="black", zorder=3)
        plt.plot(x, latencies, color="#888", linestyle="--", alpha=0.6, zorder=2)

        # Stage divider lines
        last_stage = None
        for i, stg in enumerate(stages):
            if stg != last_stage and i > 0:
                plt.axvline(i - 0.5, color="gray", linestyle=":", alpha=0.5)
            last_stage = stg

        plt.xlabel("Sequential Request Number")
        plt.ylabel("Response Time (ms)")
        plt.title("Phase 15: Failure Injection & Recovery Latency Timeline (Green=200 OK, Red=502/503)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_PLOTS_DIR, "01_failure_injection_timeline.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info("Saved failure injection plot to %s", plot_path)


class DeploymentSanityEvaluator:
    """Executes a lightweight deployment sanity check measuring performance under nominal vs fault states."""

    def __init__(self, lb_port: int = 8000, server_ports: Optional[List[int]] = None):
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.backends = [f"http://127.0.0.1:{p}" for p in self.server_ports]
        self.cluster = DeploymentFaultRunner(lb_port=lb_port, server_ports=server_ports)

    def run_sanity_evaluation(self) -> Dict[str, Any]:
        logger.info("Executing Phase 15 Deployment Sanity Evaluation...")
        self.cluster.start_all()

        sanity_results: Dict[str, Any] = {}
        try:
            # 1. Nominal State (3 backends active, 100 requests)
            t_start = time.perf_counter()
            nominal_latencies = []
            nominal_success = 0
            for _ in range(80):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.01")
                nominal_latencies.append(lat)
                if st == 200:
                    nominal_success += 1
            t_duration = time.perf_counter() - t_start

            sanity_results["nominal_3_nodes"] = {
                "total_requests": 80,
                "successful_requests": nominal_success,
                "success_rate_pct": round((nominal_success / 80) * 100.0, 2),
                "duration_seconds": round(t_duration, 3),
                "throughput_rps": round(nominal_success / max(0.01, t_duration), 2),
                "median_latency_ms": round(float(np.median(nominal_latencies)), 2),
                "p95_latency_ms": round(float(np.percentile(nominal_latencies, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(nominal_latencies, 99)), 2),
            }

            # 2. Degraded State (Server-1 down, 50 requests)
            self.cluster.stop_backend(self.server_ports[0])
            t_start = time.perf_counter()
            degraded_latencies = []
            degraded_success = 0
            for _ in range(60):
                st, srv, lat = make_request(f"http://127.0.0.1:{self.lb_port}/process?duration=0.01")
                degraded_latencies.append(lat)
                if st == 200:
                    degraded_success += 1
            t_duration = time.perf_counter() - t_start

            sanity_results["degraded_2_nodes"] = {
                "total_requests": 60,
                "successful_requests": degraded_success,
                "success_rate_pct": round((degraded_success / 60) * 100.0, 2),
                "duration_seconds": round(t_duration, 3),
                "throughput_rps": round(degraded_success / max(0.01, t_duration), 2),
                "median_latency_ms": round(float(np.median(degraded_latencies)), 2),
                "p95_latency_ms": round(float(np.percentile(degraded_latencies, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(degraded_latencies, 99)), 2),
            }

        finally:
            self.cluster.stop_all()

        with open(SANITY_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(sanity_results, f, indent=2)
        logger.info("Saved deployment sanity results to %s", SANITY_JSON_PATH)
        return sanity_results


if __name__ == "__main__":
    fault_runner = DeploymentFaultRunner()
    fault_runner.run_fault_tests()

    sanity_eval = DeploymentSanityEvaluator()
    sanity_eval.run_sanity_evaluation()
