"""Feature extraction for ML-driven load balancing.

Converts real-time ServerMetrics into deterministic 15-feature vectors
matching the model training schema.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from monitoring.collector import ServerMetrics
from config.backends import DEFAULT_BACKENDS
from ml.evaluation import ACTIVE_PRE_ROUTING_FEATURES

# The 5 active metric attributes per backend server (queue_length is zero-variance and excluded)
METRIC_FIELDS = [
    "cpu_percent",
    "memory_percent",
    "active_connections",
    "avg_response_time_ms",
    "network_latency_ms",
]

DEFAULT_SERVER_URL_MAP = {
    "server-1": "http://127.0.0.1:8001",
    "server-2": "http://127.0.0.1:8002",
    "server-3": "http://127.0.0.1:8003",
}

DEFAULT_URL_SERVER_MAP = {v: k for k, v in DEFAULT_SERVER_URL_MAP.items()}


def extract_features_from_metrics(
    metrics_by_backend: Dict[str, ServerMetrics],
    backends: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """Extract ordered 15-feature DataFrame and backend availability map.

    Args:
        metrics_by_backend: Mapping from backend URL to ServerMetrics object.
        backends: Optional list of backend URLs. If None or empty, DEFAULT_BACKENDS is used.

    Returns:
        (df_features, availability_map):
            df_features: 1-row pandas DataFrame with columns matching ACTIVE_PRE_ROUTING_FEATURES.
            availability_map: Mapping from server label (e.g. 'server-1') and backend URL to bool.
    """
    if backends:
        target_backends = sorted(backends)
    elif metrics_by_backend:
        target_backends = sorted(metrics_by_backend.keys())
    else:
        target_backends = sorted(DEFAULT_BACKENDS)

    features_dict: Dict[str, float] = {}
    availability_map: Dict[str, bool] = {}

    for idx, backend_url in enumerate(target_backends[:3], start=1):
        server_key = f"server-{idx}"
        metric = metrics_by_backend.get(backend_url)

        if metric is not None and getattr(metric, "available", True):
            is_available = True
            cpu = float(getattr(metric, "cpu_percent", 0.0))
            mem = float(getattr(metric, "memory_percent", 0.0))
            conn = float(getattr(metric, "active_connections", 0.0))
            resp_time = float(getattr(metric, "avg_response_time_ms", 0.0))
            lat = float(getattr(metric, "network_latency_ms", 0.0))
        else:
            is_available = False
            cpu = 0.0
            mem = 0.0
            conn = 0.0
            resp_time = 0.0
            lat = 0.0

        availability_map[server_key] = is_available
        availability_map[backend_url] = is_available

        features_dict[f"server_{idx}_cpu"] = cpu
        features_dict[f"server_{idx}_memory"] = mem
        features_dict[f"server_{idx}_connections"] = conn
        features_dict[f"server_{idx}_response_time"] = resp_time
        features_dict[f"server_{idx}_network_latency"] = lat

    df = pd.DataFrame([features_dict], columns=ACTIVE_PRE_ROUTING_FEATURES)
    return df, availability_map

