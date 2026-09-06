"""Experiment Runner for orchestrating automated experimental data collection."""

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from client.workload_generator import WorkloadConfig, WorkloadGenerator, get_scenario
from experiments.recorder import ExperimentRecorder
from experiments.validator import DataValidator
from load_balancer.app import create_load_balancer
from server.app import create_server

logger = logging.getLogger("experiments.runner")


class ExperimentRunner:
    """Orchestrates reproducible data collection experiments across scenarios and algorithms."""

    def __init__(
        self,
        lb_host: str = "127.0.0.1",
        lb_port: int = 8000,
        server_ports: Optional[List[int]] = None,
        data_dir: str = "data",
    ):
        self.lb_host = lb_host
        self.lb_port = lb_port
        self.server_ports = server_ports or [8001, 8002, 8003]
        self.backends = [f"http://{lb_host}:{p}" for p in self.server_ports]
        self.recorder = ExperimentRecorder(base_dir=data_dir)
        self.all_metadata: List[Dict[str, Any]] = []
        self.all_records: List[Any] = []

    def run_experiment(
        self,
        scenario: str,
        routing_algorithm: str,
        run_number: int = 1,
        num_requests: Optional[int] = None,
        concurrency: Optional[int] = None,
        seed: Optional[int] = 42,
        load_balancer_instance: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute a single controlled experiment and record all pre-routing observations."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_{scenario}_{routing_algorithm}_r{run_number}_{timestamp_str}"
        target_url = f"http://{self.lb_host}:{self.lb_port}"

        # Build workload config
        overrides: Dict[str, Any] = {"experiment_id": experiment_id}
        if num_requests is not None:
            overrides["num_requests"] = num_requests
        if concurrency is not None:
            overrides["concurrency"] = concurrency
        if seed is not None:
            overrides["seed"] = seed

        workload_cfg: WorkloadConfig = get_scenario(scenario, target_url=target_url, **overrides)

        metadata: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "scenario": scenario,
            "run_number": run_number,
            "routing_algorithm": routing_algorithm,
            "seed": seed,
            "concurrency": workload_cfg.concurrency,
            "num_requests": workload_cfg.num_requests,
            "request_rate": workload_cfg.request_rate,
            "request_duration": workload_cfg.request_duration,
            "target_url": target_url,
            "backends": self.backends,
            "software_version": "1.0.0-phase5",
        }

        # Start recorder
        self.recorder.start_experiment(
            experiment_id=experiment_id,
            scenario=scenario,
            routing_algorithm=routing_algorithm,
            metadata=metadata,
        )

        # Attach recorder to load balancer if provided
        if load_balancer_instance:
            load_balancer_instance.recorder = self.recorder

        # Execute workload
        gen = WorkloadGenerator(workload_cfg)
        gen.run()

        # Finalize and export
        records = self.recorder.stop_experiment()
        raw_csv = self.recorder.export_raw(experiment_id)
        processed_csv = self.recorder.export_processed(experiment_id)

        self.all_metadata.append(self.recorder.metadata)
        self.all_records.extend(records)

        # Validate
        val_result = DataValidator.validate_dataset([r.to_dict() for r in records])

        return {
            "experiment_id": experiment_id,
            "scenario": scenario,
            "routing_algorithm": routing_algorithm,
            "run_number": run_number,
            "total_records": len(records),
            "valid_records": val_result["valid_records"],
            "raw_csv": str(raw_csv),
            "processed_csv": str(processed_csv),
            "validation": val_result,
        }

    def generate_overall_quality_report(self) -> Dict[str, Any]:
        """Generate combined quality report across all completed experiments in runner."""
        all_records = [r.to_dict() for r in self.all_records]
        return DataValidator.generate_quality_report(all_records, self.all_metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental Data Collection Runner")
    parser.add_argument("--scenario", default="low_traffic", help="Workload scenario name")
    parser.add_argument("--algorithm", default="round_robin", choices=["round_robin", "least_connections", "ip_hash"], help="Routing algorithm")
    parser.add_argument("--requests", type=int, default=None, help="Override number of requests")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency")
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--seed", type=int, default=42, help="Starting random seed")
    parser.add_argument("--data-dir", default="data", help="Output base directory")
    args = parser.parse_args()

    runner = ExperimentRunner(data_dir=args.data_dir)
    print(f"Starting experimental data collection for {args.scenario} under {args.algorithm}...")

    results = []
    for r in range(1, args.runs + 1):
        res = runner.run_experiment(
            scenario=args.scenario,
            routing_algorithm=args.algorithm,
            run_number=r,
            num_requests=args.requests,
            concurrency=args.concurrency,
            seed=args.seed + r - 1,
        )
        results.append(res)
        print(f"Completed run {r}/{args.runs}: {res['total_records']} observations -> {res['raw_csv']}")

    report = runner.generate_overall_quality_report()
    print("\nData Quality Report:")
    print(json.dumps(report, indent=2))
