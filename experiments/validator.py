"""Validation and Data Quality reporting for experimental datasets."""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set

REQUIRED_COLUMNS = [
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
]


class DataValidator:
    """Validates experimental observation records and generates data quality reports."""

    @staticmethod
    def validate_record(record: Dict[str, Any]) -> List[str]:
        """Validate a single record against schema, range, and temporal constraints."""
        errors: List[str] = []

        # 1. Required columns presence
        for col in REQUIRED_COLUMNS:
            if col not in record:
                errors.append(f"Missing required column: '{col}'")

        # 2. Check identifier validity
        if not str(record.get("experiment_id", "")).strip():
            errors.append("Invalid or empty experiment_id")
        try:
            req_id = int(record.get("request_id", 0))
            if req_id <= 0:
                errors.append(f"Non-positive request_id: {req_id}")
        except (ValueError, TypeError):
            errors.append("request_id must be an integer")

        # 3. Numeric ranges for server metrics
        for s in (1, 2, 3):
            # CPU (0.0 to 100.0)
            cpu = record.get(f"server_{s}_cpu")
            if cpu is None or not isinstance(cpu, (int, float)) or not (0.0 <= cpu <= 100.0):
                errors.append(f"server_{s}_cpu out of range [0, 100]: {cpu}")

            # Memory (0.0 to 100.0)
            mem = record.get(f"server_{s}_memory")
            if mem is None or not isinstance(mem, (int, float)) or not (0.0 <= mem <= 100.0):
                errors.append(f"server_{s}_memory out of range [0, 100]: {mem}")

            # Active connections (>= 0)
            conn = record.get(f"server_{s}_connections")
            if conn is None or not isinstance(conn, int) or conn < 0:
                errors.append(f"server_{s}_connections must be non-negative integer: {conn}")

            # Queue length (>= 0)
            queue_len = record.get(f"server_{s}_queue_length")
            if queue_len is None or not isinstance(queue_len, int) or queue_len < 0:
                errors.append(f"server_{s}_queue_length must be non-negative integer: {queue_len}")

            # Latency (>= 0.0)
            lat = record.get(f"server_{s}_network_latency")
            if lat is None or not isinstance(lat, (int, float)) or lat < 0.0:
                errors.append(f"server_{s}_network_latency must be non-negative: {lat}")

            # Response time (>= 0.0)
            resp_t = record.get(f"server_{s}_response_time")
            if resp_t is None or not isinstance(resp_t, (int, float)) or resp_t < 0.0:
                errors.append(f"server_{s}_response_time must be non-negative: {resp_t}")

        # 4. Actual response time
        actual_rt = record.get("actual_response_time")
        if actual_rt is None or not isinstance(actual_rt, (int, float)) or actual_rt < 0.0:
            errors.append(f"actual_response_time must be non-negative: {actual_rt}")

        # 5. Temporal consistency
        ts = record.get("timestamp")
        req_start = record.get("request_start")
        req_end = record.get("request_end")
        if req_start and ts and (req_start < ts - 0.05):
            errors.append(f"Temporal inconsistency: request_start ({req_start}) < timestamp ({ts})")
        if req_start and req_end and (req_end < req_start):
            errors.append(f"Temporal inconsistency: request_end ({req_end}) < request_start ({req_start})")

        return errors

    @classmethod
    def validate_dataset(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a collection of records and check for duplicates and missing values."""
        results: Dict[str, Any] = {
            "total_records": len(records),
            "valid_records": 0,
            "invalid_records": 0,
            "duplicate_records": 0,
            "missing_values": {},
            "errors": [],
        }

        seen_keys: Set[tuple] = set()
        missing_counter: Counter = Counter()

        for idx, rec in enumerate(records):
            # Check duplicate (experiment_id, request_id)
            key = (rec.get("experiment_id"), rec.get("request_id"))
            if key in seen_keys:
                results["duplicate_records"] += 1
                results["errors"].append(f"Record {idx}: duplicate (experiment_id, request_id)={key}")
            seen_keys.add(key)

            # Check missing fields for non-optional columns
            for col in REQUIRED_COLUMNS:
                if col in ("request_rate", "best_server"):
                    continue
                val = rec.get(col)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    missing_counter[col] += 1

            # Validate individual constraints
            rec_errors = cls.validate_record(rec)
            if rec_errors:
                results["invalid_records"] += 1
                results["errors"].extend([f"Record {idx}: {e}" for e in rec_errors[:3]])
            else:
                results["valid_records"] += 1

        results["missing_values"] = dict(missing_counter)
        return results

    @classmethod
    def generate_quality_report(
        cls,
        records: List[Dict[str, Any]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Produce a comprehensive data quality report without ML evaluation metrics."""
        validation = cls.validate_dataset(records)
        total = len(records)

        scenarios = Counter(r.get("workload_scenario", "unknown") for r in records)
        algorithms = Counter(r.get("routing_algorithm", "unknown") for r in records)
        experiments = Counter(r.get("experiment_id", "unknown") for r in records)

        failed_requests = sum(1 for r in records if not r.get("request_success", True))
        labels = Counter(str(r.get("best_server")) for r in records)

        # Feature range calculations for key server metrics
        feature_ranges: Dict[str, Dict[str, float]] = {}
        metric_cols = [
            "server_1_cpu", "server_1_memory", "server_1_connections", "server_1_response_time", "server_1_network_latency", "server_1_queue_length",
            "server_2_cpu", "server_2_memory", "server_2_connections", "server_2_response_time", "server_2_network_latency", "server_2_queue_length",
            "server_3_cpu", "server_3_memory", "server_3_connections", "server_3_response_time", "server_3_network_latency", "server_3_queue_length",
            "actual_response_time",
        ]

        if total > 0:
            for col in metric_cols:
                vals = [float(r[col]) for r in records if r.get(col) is not None and isinstance(r.get(col), (int, float))]
                if vals:
                    feature_ranges[col] = {
                        "min": round(min(vals), 3),
                        "max": round(max(vals), 3),
                        "mean": round(sum(vals) / len(vals), 3),
                    }

        return {
            "total_experiments": len(experiments) if not metadata_list else len(metadata_list),
            "total_observations": total,
            "valid_observations": validation["valid_records"],
            "invalid_observations": validation["invalid_records"],
            "duplicate_records": validation["duplicate_records"],
            "missing_values": validation["missing_values"],
            "failed_requests": failed_requests,
            "failed_request_rate_percent": round((failed_requests / total * 100.0) if total > 0 else 0.0, 2),
            "observations_per_scenario": dict(scenarios),
            "observations_per_routing_algorithm": dict(algorithms),
            "label_distribution": dict(labels),
            "feature_ranges": feature_ranges,
        }
