"""Routing strategies and abstraction for the Load Balancer."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import hashlib
import threading
from typing import Dict, List, Optional
import time
from typing import Dict, List, Optional, Any
import numpy as np


class BaseRouter(ABC):
    """Abstract base class for load balancing routing strategies."""

    def __init__(self, backends: List[str]):
        if not backends:
            raise ValueError("Backends list cannot be empty")
        self.backends = list(backends)

    @abstractmethod
    def select(self, client_ip: Optional[str] = None) -> str:
        """Select a backend server from the list of backends."""
        pass

    def release(self, backend: str, success: bool = True) -> None:
        """Release backend after request completion.

        Override in stateful routers (e.g. LeastConnections).
        """
        pass

    @contextmanager
    def route(self, client_ip: Optional[str] = None):
        """Context manager to acquire a backend and guarantee release on completion or failure."""
        backend = self.select(client_ip=client_ip)
        success = False
        try:
            yield backend
            success = True
        finally:
            self.release(backend, success=success)


class RoundRobinRouter(BaseRouter):
    """Round-robin routing strategy.

    Sequentially cycles through the list of backend servers.
    Thread-safe using a threading.Lock.
    """

    def __init__(self, backends: List[str]):
        super().__init__(backends)
        self._index = 0
        self._lock = threading.Lock()

    def select(self, client_ip: Optional[str] = None) -> str:
        with self._lock:
            selected = self.backends[self._index % len(self.backends)]
            self._index = (self._index + 1) % len(self.backends)
            return selected


class LeastConnectionsRouter(BaseRouter):
    """Least-connections routing strategy.

    Routes incoming requests to the backend server with the lowest number of
    currently active connections.

    Tie-breaking policy:
    When multiple backend servers share the identical minimum active connection count,
    the server that appears earliest in the configured backends list (lowest index)
    is chosen deterministically.

    Note:
    Connection counts here are internal routing state used exclusively for server
    selection and do not represent the Phase 3 real-time server monitoring metrics.
    """

    def __init__(self, backends: List[str]):
        super().__init__(backends)
        self._active_connections: Dict[str, int] = {b: 0 for b in self.backends}
        self._lock = threading.Lock()

    def get_active_connections(self) -> Dict[str, int]:
        """Return a copy of the current active connection counts."""
        with self._lock:
            return dict(self._active_connections)

    def set_active_connections(self, backend: str, count: int) -> None:
        """Manually update active connection count for a backend (useful for testing)."""
        with self._lock:
            if backend in self._active_connections:
                self._active_connections[backend] = count

    def select(self, client_ip: Optional[str] = None) -> str:
        with self._lock:
            # Deterministic selection: min by active count.
            # Python's min is stable: on equal keys, the first element encountered is returned.
            selected = min(self.backends, key=lambda b: self._active_connections[b])
            self._active_connections[selected] += 1
            return selected

    def release(self, backend: str, success: bool = True) -> None:
        with self._lock:
            if backend in self._active_connections:
                self._active_connections[backend] = max(0, self._active_connections[backend] - 1)


class IPHashRouter(BaseRouter):
    """IP-hash routing strategy.

    Deterministically maps a client IP to a backend server using a stable cryptographic hash.

    Hashing policy:
    1. Extract client IP (trimmed string). If missing or empty, fall back to "127.0.0.1".
    2. Compute MD5 digest from the UTF-8 bytes of the IP string using hashlib.md5.
    3. Convert the hexadecimal digest to an integer and compute modulo len(backends).
    4. Return the backend at that computed index.
    """

    def select(self, client_ip: Optional[str] = None) -> str:
        ip = (client_ip or "").strip()
        if not ip:
            ip = "127.0.0.1"

        digest = hashlib.md5(ip.encode("utf-8")).hexdigest()
        index = int(digest, 16) % len(self.backends)
        return self.backends[index]


class MLRouter(BaseRouter):
    """Machine-learning driven routing strategy with deterministic safety fallback.

    Uses real-time backend server metrics and a trained model pipeline to predict
    the optimal backend server for incoming requests.

    Pluggable design:
    - Accepts any trained model pipeline (LogisticRegression, RandomForest, etc.)
      or loads from a joblib file path.
    - Strictly adheres to the 15 active pre-routing feature contract.
    - Verifies backend health and availability.
    - If model artifact is unavailable, inference fails, feature vector is malformed,
      or the predicted backend is unreachable/unhealthy, safely and deterministically
      falls back to LeastConnectionsRouter among healthy backends.
    - Exposes prediction metadata (predicted server, confidence score, fallback flag, latency).
    """

    DEFAULT_MODEL_PATH = "models/logistic_regression.joblib"

    def __init__(
        self,
        backends: List[str],
        model: Optional[Any] = None,
        model_path: Optional[str] = None,
        collector: Optional[Any] = None,
        fallback_router: Optional[BaseRouter] = None,
    ):
        super().__init__(backends)
        self.collector = collector
        self.fallback_router = fallback_router or LeastConnectionsRouter(backends)
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = model
        self.last_prediction: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Map backends by port/order to server-1, server-2, server-3
        sorted_backends = sorted(self.backends)
        self.backend_to_label = {b: f"server-{i}" for i, b in enumerate(sorted_backends[:3], start=1)}
        self.label_to_backend = {f"server-{i}": b for i, b in enumerate(sorted_backends[:3], start=1)}
        self.id_to_label = {0: "server-1", 1: "server-2", 2: "server-3"}

        # Attempt initial model load if model not passed directly
        if self.model is None:
            self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the model pipeline from model_path."""
        try:
            import os
            import joblib
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            else:
                self.model = None
        except Exception:
            self.model = None

    def select(self, client_ip: Optional[str] = None) -> str:
        """Predict the best backend server or fall back to LeastConnectionsRouter."""
        inference_start = time.perf_counter()
        with self._lock:
            # Check model availability
            if self.model is None:
                self._load_model()

            if self.model is None or self.collector is None:
                selected = self.fallback_router.select(client_ip=client_ip)
                inference_ms = round((time.perf_counter() - inference_start) * 1000.0, 3)
                self.last_prediction = {
                    "predicted_server": None,
                    "confidence": None,
                    "is_fallback": True,
                    "fallback_reason": "Model or collector unavailable",
                    "chosen_server": selected,
                    "inference_latency_ms": inference_ms,
                }
                return selected

            try:
                # 1. Collect real-time metrics
                metrics_by_backend = self.collector.collect_all()

                # 2. Extract features and availability
                from ml.features import extract_features_from_metrics
                df_features, availability_map = extract_features_from_metrics(
                    metrics_by_backend, backends=self.backends
                )

                # Determine which backends are actually available
                healthy_backends = [b for b in self.backends if availability_map.get(b, True)]
                if not healthy_backends:
                    # Fallback if no backend reported available
                    selected = self.fallback_router.select(client_ip=client_ip)
                    inference_ms = round((time.perf_counter() - inference_start) * 1000.0, 3)
                    self.last_prediction = {
                        "predicted_server": None,
                        "confidence": None,
                        "is_fallback": True,
                        "fallback_reason": "No backends healthy",
                        "chosen_server": selected,
                        "inference_latency_ms": inference_ms,
                    }
                    return selected

                # 3. Model inference
                pred = self.model.predict(df_features)
                raw_pred = pred[0]
                if isinstance(raw_pred, (int, np.integer)):
                    pred_label = self.id_to_label.get(int(raw_pred), f"server-{raw_pred + 1}")
                else:
                    pred_label = str(raw_pred)

                confidence = None
                if hasattr(self.model, "predict_proba"):
                    try:
                        probs = self.model.predict_proba(df_features)[0]
                        if isinstance(raw_pred, (int, np.integer)) and int(raw_pred) < len(probs):
                            confidence = float(probs[int(raw_pred)])
                        else:
                            confidence = float(np.max(probs))
                    except Exception:
                        confidence = None

                predicted_backend = self.label_to_backend.get(pred_label)

                # 4. Health & Safety Validation
                if predicted_backend and predicted_backend in healthy_backends:
                    chosen = predicted_backend
                    is_fallback = False
                    fallback_reason = None
                else:
                    # Predicted backend is offline or invalid -> fallback
                    chosen = self.fallback_router.select(client_ip=client_ip)
                    is_fallback = True
                    fallback_reason = (
                        f"Predicted backend {pred_label} ({predicted_backend}) is unavailable"
                        if predicted_backend
                        else f"Unknown predicted label {pred_label}"
                    )

                inference_ms = round((time.perf_counter() - inference_start) * 1000.0, 3)
                self.last_prediction = {
                    "predicted_server": predicted_backend or pred_label,
                    "confidence": round(confidence, 4) if confidence is not None else None,
                    "is_fallback": is_fallback,
                    "fallback_reason": fallback_reason,
                    "chosen_server": chosen,
                    "inference_latency_ms": inference_ms,
                }
                return chosen

            except Exception as e:
                selected = self.fallback_router.select(client_ip=client_ip)
                inference_ms = round((time.perf_counter() - inference_start) * 1000.0, 3)
                self.last_prediction = {
                    "predicted_server": None,
                    "confidence": None,
                    "is_fallback": True,
                    "fallback_reason": f"Inference exception: {str(e)}",
                    "chosen_server": selected,
                    "inference_latency_ms": inference_ms,
                }
                return selected

    def release(self, backend: str, success: bool = True) -> None:
        """Release backend on fallback router if it tracks active connections."""
        if hasattr(self.fallback_router, "release"):
            self.fallback_router.release(backend, success=success)


ROUTER_REGISTRY = {
    "round_robin": RoundRobinRouter,
    "least_connections": LeastConnectionsRouter,
    "ip_hash": IPHashRouter,
    "ml": MLRouter,
}


def get_router(
    algorithm: str,
    backends: List[str],
    collector: Optional[Any] = None,
    model_path: Optional[str] = None,
) -> BaseRouter:
    """Factory to instantiate a router by name."""
    normalized = algorithm.lower().replace("-", "_").strip()
    if normalized not in ROUTER_REGISTRY:
        supported = list(ROUTER_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {supported}")
    router_cls = ROUTER_REGISTRY[normalized]
    if normalized == "ml":
        return router_cls(backends=backends, collector=collector, model_path=model_path)
    return router_cls(backends=backends)


