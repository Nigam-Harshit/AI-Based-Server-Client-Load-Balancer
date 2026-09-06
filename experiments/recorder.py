"""Experiment Recorder for capturing pre-routing cluster state, outcomes, and labels."""

import csv
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from monitoring.collector import MetricsCollector, ServerMetrics

logger = logging.getLogger("experiments.recorder")

RECORD_FIELDNAMES = [
    "experiment_id",
    "timestamp",
    "request_id",
    "workload_scenario",
    "request_type",
    "request_size",
    "concurrency",
    "request_rate",
    "routing_algorithm",
    "server_1_cpu",
    "server_1_memory",
    "server_1_connections",
    "server_1_response_time",
    "server_1_network_latency",
    "server_1_queue_length",
    "server_2_cpu",
    "server_2_memory",
    "server_2_connections",
    "server_2_response_time",
    "server_2_network_latency",
    "server_2_queue_length",
    "server_3_cpu",
    "server_3_memory",
    "server_3_connections",
    "server_3_response_time",
    "server_3_network_latency",
    "server_3_queue_length",
    "selected_server",
    "actual_response_time",
    "request_success",
    "best_server",
    "request_start",
    "request_end",
]


@dataclass
class ExperimentRecord:
    """Represents a single routing observation with pre-state, outcome, and label."""

    experiment_id: str
    timestamp: float
    request_id: int
    workload_scenario: str
    request_type: str
    request_size: int
    concurrency: int
    request_rate: Optional[float]
    routing_algorithm: str

    server_1_cpu: float
    server_1_memory: float
    server_1_connections: int
    server_1_response_time: float
    server_1_network_latency: float
    server_1_queue_length: int

    server_2_cpu: float
    server_2_memory: float
    server_2_connections: int
    server_2_response_time: float
    server_2_network_latency: float
    server_2_queue_length: int

    server_3_cpu: float
    server_3_memory: float
    server_3_connections: int
    server_3_response_time: float
    server_3_network_latency: float
    server_3_queue_length: int

    selected_server: str
    actual_response_time: float
    request_success: bool

    best_server: Optional[str] = None
    request_start: Optional[float] = None
    request_end: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRecorder:
    """Manages recording of pre-routing metrics and outcomes for experiments."""

    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self.is_recording = False
        self.current_experiment_id: Optional[str] = None
        self.current_scenario: Optional[str] = None
        self.current_algorithm: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.records: List[ExperimentRecord] = []
        self._request_counter = 0

    def start_experiment(
        self,
        experiment_id: str,
        scenario: str,
        routing_algorithm: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Begin recording observations for a given experiment."""
        with self._lock:
            self.current_experiment_id = experiment_id
            self.current_scenario = scenario
            self.current_algorithm = routing_algorithm
            self.metadata = metadata or {}
            self.metadata["experiment_id"] = experiment_id
            self.metadata["scenario"] = scenario
            self.metadata["routing_algorithm"] = routing_algorithm
            self.metadata["start_time"] = time.time()
            self.records = []
            self._request_counter = 0
            self.is_recording = True
            logger.info("Started recording experiment: %s", experiment_id)

    def stop_experiment(self) -> List[ExperimentRecord]:
        """Stop recording and finalize labels for the completed experiment."""
        with self._lock:
            self.is_recording = False
            self.metadata["end_time"] = time.time()
            self.metadata["total_records"] = len(self.records)

            # Assign labels to all records
            for record in self.records:
                record.best_server = self.compute_label(record)

            logger.info(
                "Stopped experiment %s with %d records",
                self.current_experiment_id,
                len(self.records),
            )
            return list(self.records)

    def capture_pre_snapshot(
        self,
        collector: MetricsCollector,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Capture real-time pre-routing metrics from all 3 servers."""
        snapshot_time = timestamp or time.time()
        all_metrics = collector.collect_all()

        # Sort backends by port (8001, 8002, 8003) for consistent mapping to 1, 2, 3
        sorted_backends = sorted(all_metrics.keys())
        snapshot: Dict[str, Any] = {"timestamp": snapshot_time}

        for idx, b_url in enumerate(sorted_backends[:3], start=1):
            m: ServerMetrics = all_metrics[b_url]
            snapshot[f"server_{idx}_cpu"] = m.cpu_percent
            snapshot[f"server_{idx}_memory"] = m.memory_percent
            snapshot[f"server_{idx}_connections"] = m.active_connections
            snapshot[f"server_{idx}_response_time"] = m.avg_response_time_ms
            snapshot[f"server_{idx}_network_latency"] = m.network_latency_ms
            snapshot[f"server_{idx}_queue_length"] = m.request_queue_length
            snapshot[f"server_{idx}_url"] = b_url
            snapshot[f"server_{idx}_available"] = m.available

        return snapshot

    def record_observation(
        self,
        pre_snapshot: Dict[str, Any],
        experiment_id: Optional[str] = None,
        request_id: Optional[Any] = None,
        workload_scenario: Optional[str] = None,
        request_type: str = "process",
        request_size: int = 0,
        concurrency: int = 1,
        request_rate: Optional[float] = None,
        routing_algorithm: Optional[str] = None,
        selected_server: str = "unknown",
        actual_response_time: float = 0.0,
        request_success: bool = True,
        request_start: Optional[float] = None,
        request_end: Optional[float] = None,
    ) -> ExperimentRecord:
        """Create and store an ExperimentRecord ensuring temporal separation."""
        with self._lock:
            self._request_counter += 1
            final_req_id = int(request_id) if request_id is not None else self._request_counter

            record = ExperimentRecord(
                experiment_id=experiment_id or self.current_experiment_id or "unknown",
                timestamp=pre_snapshot.get("timestamp", time.time()),
                request_id=final_req_id,
                workload_scenario=workload_scenario or self.current_scenario or "unknown",
                request_type=request_type,
                request_size=request_size,
                concurrency=concurrency,
                request_rate=request_rate,
                routing_algorithm=routing_algorithm or self.current_algorithm or "unknown",
                server_1_cpu=float(pre_snapshot.get("server_1_cpu", 0.0)),
                server_1_memory=float(pre_snapshot.get("server_1_memory", 0.0)),
                server_1_connections=int(pre_snapshot.get("server_1_connections", 0)),
                server_1_response_time=float(pre_snapshot.get("server_1_response_time", 0.0)),
                server_1_network_latency=float(pre_snapshot.get("server_1_network_latency", 0.0)),
                server_1_queue_length=int(pre_snapshot.get("server_1_queue_length", 0)),
                server_2_cpu=float(pre_snapshot.get("server_2_cpu", 0.0)),
                server_2_memory=float(pre_snapshot.get("server_2_memory", 0.0)),
                server_2_connections=int(pre_snapshot.get("server_2_connections", 0)),
                server_2_response_time=float(pre_snapshot.get("server_2_response_time", 0.0)),
                server_2_network_latency=float(pre_snapshot.get("server_2_network_latency", 0.0)),
                server_2_queue_length=int(pre_snapshot.get("server_2_queue_length", 0)),
                server_3_cpu=float(pre_snapshot.get("server_3_cpu", 0.0)),
                server_3_memory=float(pre_snapshot.get("server_3_memory", 0.0)),
                server_3_connections=int(pre_snapshot.get("server_3_connections", 0)),
                server_3_response_time=float(pre_snapshot.get("server_3_response_time", 0.0)),
                server_3_network_latency=float(pre_snapshot.get("server_3_network_latency", 0.0)),
                server_3_queue_length=int(pre_snapshot.get("server_3_queue_length", 0)),
                selected_server=selected_server,
                actual_response_time=actual_response_time,
                request_success=request_success,
                best_server=None,
                request_start=request_start,
                request_end=request_end,
            )

            record.best_server = self.compute_label(record)
            self.records.append(record)
            return record

    def compute_label(self, record: ExperimentRecord) -> Optional[str]:
        """Compute the optimal server label independently of selected_server.

        Policy:
        Cost(S_i) = network_latency_i + response_time_i * (1 + connections_i + queue_length_i)
        If all costs are within 2.0 ms of each other (idle cluster tie), mark as None.
        Otherwise, return the server identifier with the minimum expected cost.
        """
        servers = [
            (
                "server-1",
                record.server_1_network_latency
                + record.server_1_response_time
                * (1 + record.server_1_connections + record.server_1_queue_length),
            ),
            (
                "server-2",
                record.server_2_network_latency
                + record.server_2_response_time
                * (1 + record.server_2_connections + record.server_2_queue_length),
            ),
            (
                "server-3",
                record.server_3_network_latency
                + record.server_3_response_time
                * (1 + record.server_3_connections + record.server_3_queue_length),
            ),
        ]

        costs = [c for _, c in servers]
        min_cost = min(costs)
        max_cost = max(costs)

        # Ambiguity / Tie handling: if difference between best and worst server is <= 2ms, mark unassigned
        if max_cost - min_cost <= 2.0 and min_cost <= 5.0:
            return None

        # Return the server with the lowest cost
        best = min(servers, key=lambda s: s[1])
        return best[0]

    def export_raw(self, experiment_id: Optional[str] = None) -> Path:
        """Export raw observations to CSV in data/raw/."""
        exp_id = experiment_id or self.current_experiment_id or f"exp_{int(time.time())}"
        out_csv = self.raw_dir / f"experiment_{exp_id}.csv"

        with self._lock:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RECORD_FIELDNAMES)
                writer.writeheader()
                for rec in self.records:
                    writer.writerow(rec.to_dict())

            # Also save raw metadata
            meta_file = self.raw_dir / f"metadata_{exp_id}.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2)

        logger.info("Exported raw experiment data to %s", out_csv)
        return out_csv

    def export_processed(self, experiment_id: Optional[str] = None) -> Path:
        """Export cleaned, validated records to data/processed/."""
        exp_id = experiment_id or self.current_experiment_id or f"exp_{int(time.time())}"
        out_csv = self.processed_dir / f"processed_{exp_id}.csv"

        with self._lock:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RECORD_FIELDNAMES)
                writer.writeheader()
                for rec in self.records:
                    writer.writerow(rec.to_dict())

        logger.info("Exported processed experiment data to %s", out_csv)
        return out_csv
