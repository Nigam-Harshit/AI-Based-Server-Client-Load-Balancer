"""Request models, priority definitions, and deadline parsing for Phase 10."""

from dataclasses import asdict, dataclass
from enum import IntEnum
from contextlib import contextmanager
import time
from typing import Any, Dict, Optional
from typing import Any, Dict, List, Optional



class PriorityLevel(IntEnum):
    """Enumeration of request priority levels ordered by urgency."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def from_str(cls, val: Optional[str]) -> "PriorityLevel":
        if not val:
            return cls.NORMAL
        norm = str(val).strip().upper()
        if norm in cls.__members__:
            return cls[norm]
        # Also support numeric inputs as strings
        try:
            num = int(norm)
            for m in cls:
                if m.value == num:
                    return m
        except ValueError:
            pass
        return cls.NORMAL

    from_string = from_str


class DeadlineUrgency(IntEnum):
    """Classification of deadline slack urgency."""
    SAFE = 1
    APPROACHING_DEADLINE = 2
    URGENT = 3
    DEADLINE_RISK = 4

    @classmethod
    def from_slack(cls, slack_ms: float) -> "DeadlineUrgency":
        """Classify urgency based on deterministic slack thresholds (in ms)."""
        if slack_ms <= 0.0:
            return cls.DEADLINE_RISK
        elif slack_ms < 50.0:
            return cls.URGENT
        elif slack_ms < 150.0:
            return cls.APPROACHING_DEADLINE
        else:
            return cls.SAFE


@dataclass
class PriorityRequestMetadata:
    """Encapsulates priority and deadline attributes for incoming requests."""
    request_id: str
    priority: PriorityLevel = PriorityLevel.NORMAL
    deadline: Optional[float] = None              # Absolute epoch timestamp deadline in seconds
    arrival_time: float = 0.0                     # Epoch arrival timestamp at load balancer
    estimated_processing_time_ms: float = 30.0   # Default heuristic estimated duration in ms
    estimated_duration_ms: Optional[float] = None # Alias for estimated_processing_time_ms

    def __post_init__(self):
        if not self.arrival_time:
            self.arrival_time = time.time()
        if isinstance(self.priority, str):
            self.priority = PriorityLevel.from_str(self.priority)
        if self.estimated_duration_ms is not None:
            self.estimated_processing_time_ms = float(self.estimated_duration_ms)
        else:
            self.estimated_duration_ms = float(self.estimated_processing_time_ms)

    def calculate_slack_ms(self, current_time: Optional[float] = None) -> Optional[float]:
        """Calculate deadline slack in milliseconds:
        slack = deadline - current_time - estimated_processing_time
        """
        if self.deadline is None:
            return None
        now = current_time or time.time()
        est_proc_sec = self.estimated_processing_time_ms / 1000.0
        slack_sec = self.deadline - now - est_proc_sec
        return round(slack_sec * 1000.0, 2)

    def classify_urgency(self, current_time: Optional[float] = None) -> DeadlineUrgency:
        """Classify urgency state based on remaining deadline slack."""
        slack = self.calculate_slack_ms(current_time)
        if slack is None:
            return DeadlineUrgency.SAFE
        return DeadlineUrgency.from_slack(slack)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "priority": self.priority.name,
            "priority_value": self.priority.value,
            "deadline": self.deadline,
            "arrival_time": self.arrival_time,
            "estimated_processing_time_ms": self.estimated_processing_time_ms,
            "slack_ms": self.calculate_slack_ms(),
            "urgency": self.classify_urgency().name,
        }

    @classmethod
    def from_headers(cls, headers: Any, request_id: str = "unknown") -> "PriorityRequestMetadata":
        """Extract priority and deadline parameters from HTTP request headers.

        Supported headers:
        - X-Request-Priority: 'LOW', 'NORMAL', 'HIGH', 'CRITICAL' (or 1, 2, 3, 4)
        - X-Request-Deadline: Absolute epoch timestamp (float) or relative seconds (e.g. '+0.5')
        - X-Estimated-Duration: Estimated processing time in seconds or ms
        - X-Arrival-Time: Original request creation timestamp
        """
        now = time.time()
        p_hdr = headers.get("X-Request-Priority") or headers.get("x-request-priority")
        d_hdr = headers.get("X-Request-Deadline") or headers.get("x-request-deadline")
        e_hdr = headers.get("X-Estimated-Duration") or headers.get("x-estimated-duration")
        a_hdr = headers.get("X-Arrival-Time") or headers.get("x-arrival-time")

        priority = PriorityLevel.from_str(p_hdr)

        arrival_time = float(a_hdr) if a_hdr else now

        deadline = None
        if d_hdr is not None:
            d_str = str(d_hdr).strip()
            try:
                if d_str.startswith("+"):
                    # Relative deadline specified in seconds from arrival
                    rel_sec = float(d_str[1:])
                    deadline = arrival_time + rel_sec
                else:
                    val = float(d_str)
                    # If val < 100000, treat as relative seconds, else absolute epoch
                    if val < 100000:
                        deadline = arrival_time + val
                    else:
                        deadline = val
            except ValueError:
                deadline = None

        est_ms = 30.0
        if e_hdr is not None:
            try:
                val = float(e_hdr)
                # If val < 10, treat as seconds, else ms
                est_ms = val * 1000.0 if val < 10.0 else val
            except ValueError:
                est_ms = 30.0

        return cls(
            request_id=str(request_id),
            priority=priority,
            deadline=deadline,
            arrival_time=arrival_time,
            estimated_processing_time_ms=est_ms,
        )


class PriorityDeadlineRouter:
    """Priority and deadline-aware routing decision layer wrapping BaseRouter.

    Implements a formal multi-factor urgency scheduling policy:
    1. Computes remaining deadline slack:
       slack = deadline - now - estimated_processing_time
    2. Classifies urgency (SAFE, APPROACHING_DEADLINE, URGENT, DEADLINE_RISK).
    3. Resolves conflicts: Deadline urgency overrides priority when slack is critical.
    4. Evaluates server stress: If the ML-predicted server has high queue/connection load
       or is degraded, evaluates expected completion time across all healthy backends:
       Expected Completion = network_latency + response_time * (1 + active_connections)
    5. Intelligently steers urgent requests to the lowest-delay healthy backend while
       routing non-urgent requests according to the underlying ML router.
    """

    def __init__(
        self,
        base_router: Any = None,
        collector: Optional[Any] = None,
        backends: Optional[list] = None,
        underlying_router: Any = None,
        metrics_collector: Optional[Any] = None,
    ):
        self.base_router = base_router or underlying_router
        self.collector = collector or metrics_collector or getattr(self.base_router, "collector", None)
        self.backends = list(backends) if backends else list(getattr(self.base_router, "backends", []))
        self.last_decision: Dict[str, Any] = {}
        self._healthy_backends: Optional[set] = None

    def set_healthy_backends(self, healthy: Optional[List[str]]) -> None:
        """Dynamically propagate healthy backends to base router and local selection."""
        if hasattr(self.base_router, "set_healthy_backends"):
            self.base_router.set_healthy_backends(healthy)
        if healthy:
            self._healthy_backends = set(healthy)
        else:
            self._healthy_backends = None

    def select(
        self,
        client_ip: Optional[str] = None,
        priority_meta: Optional[PriorityRequestMetadata] = None,
    ) -> str:
        """Route request by combining ML prediction with priority/deadline evaluation."""
        now = time.time()
        meta = priority_meta or PriorityRequestMetadata(request_id="unknown")
        slack_ms = meta.calculate_slack_ms(now)
        urgency = meta.classify_urgency(now)

        # 1. Obtain underlying base router (ML) prediction
        ml_predicted = self.base_router.select(client_ip=client_ip)
        ml_pred_meta = getattr(self.base_router, "last_prediction", {})
        raw_ml_pred = ml_pred_meta.get("predicted_server") or ml_predicted

        # 2. Collect real-time metrics for backend candidate evaluation
        if self.collector:
            if hasattr(self.collector, "collect_all"):
                metrics_by_backend = self.collector.collect_all()
            elif hasattr(self.collector, "get_latest_metrics"):
                metrics_by_backend = self.collector.get_latest_metrics()
            else:
                metrics_by_backend = {}
        else:
            metrics_by_backend = {}
        healthy_backends = [
            b for b in self.backends
            if metrics_by_backend.get(b) is None or getattr(metrics_by_backend[b], "available", True)
        ]
        if self._healthy_backends is not None:
            healthy_backends = [b for b in healthy_backends if b in self._healthy_backends]
        if not healthy_backends:
            healthy_backends = list(self.backends)
            if self._healthy_backends is not None:
                healthy_backends = [b for b in self.backends if b in self._healthy_backends]
            else:
                healthy_backends = list(self.backends)

        # 3. Calculate expected turnaround latency for each healthy backend:
        # Expected Turnaround = network_latency + avg_response_time * (1 + active_connections)
        backend_expected_lat: Dict[str, float] = {}
        for b in healthy_backends:
            m = metrics_by_backend.get(b)
            if m and getattr(m, "available", True):
                lat = float(getattr(m, "network_latency_ms", 1.0))
                resp = float(getattr(m, "avg_response_time_ms", 25.0))
                conn = int(getattr(m, "active_connections", 0))
                backend_expected_lat[b] = lat + resp * (1.0 + conn)
            else:
                backend_expected_lat[b] = 9999.0

        fastest_backend = min(healthy_backends, key=lambda b: backend_expected_lat.get(b, 9999.0))
        ml_expected_lat = backend_expected_lat.get(ml_predicted, 9999.0)

        # 4. Priority & Deadline Decision Policy
        final_backend = ml_predicted
        priority_override = False
        deadline_override = False
        routing_reason = "normal_ml_decision"

        # Check if ML predicted an unhealthy backend
        if ml_predicted not in healthy_backends:
            final_backend = fastest_backend
            routing_reason = "unhealthy_backend_fallback"

        # Condition 1: Imminent Deadline Risk or Extreme Urgency
        elif urgency in (DeadlineUrgency.DEADLINE_RISK, DeadlineUrgency.URGENT):
            # If ML backend turnaround is worse than fastest available backend by > 15ms, override
            if (ml_expected_lat - backend_expected_lat[fastest_backend]) > 15.0 or (slack_ms is not None and slack_ms < ml_expected_lat):
                final_backend = fastest_backend
                deadline_override = True
                routing_reason = "deadline_slack_override"

        # Condition 2: High or Critical Priority Contention
        elif meta.priority in (PriorityLevel.HIGH, PriorityLevel.CRITICAL):
            # For critical priority, if ML backend is carrying connection load > 0 while another healthy backend is completely idle
            try:
                ml_conn = int(getattr(metrics_by_backend.get(ml_predicted), "active_connections", 0))
            except Exception:
                ml_conn = 0
            try:
                fastest_conn = int(getattr(metrics_by_backend.get(fastest_backend), "active_connections", 0))
            except Exception:
                fastest_conn = 0
            if ml_conn > fastest_conn and (ml_expected_lat - backend_expected_lat[fastest_backend]) > 10.0:
                final_backend = fastest_backend
                priority_override = True
                routing_reason = "priority_load_override"

        # Record decision observability metadata
        self.last_decision = {
            "request_id": meta.request_id,
            "priority": meta.priority.name,
            "priority_val": meta.priority.value,
            "deadline": meta.deadline,
            "slack_ms": slack_ms,
            "urgency": urgency.name,
            "ml_predicted_server": raw_ml_pred,
            "final_backend": final_backend,
            "priority_override": priority_override,
            "deadline_override": deadline_override,
            "routing_reason": routing_reason,
            "ml_expected_lat_ms": round(ml_expected_lat, 2) if ml_expected_lat != 9999.0 else None,
            "best_expected_lat_ms": round(backend_expected_lat[fastest_backend], 2) if backend_expected_lat else None,
        }

        return final_backend

    def release(self, backend: str, success: bool = True) -> None:
        """Forward release notification to base router."""
        if hasattr(self.base_router, "release"):
            self.base_router.release(backend, success=success)

    @contextmanager
    def route(
        self,
        client_ip: Optional[str] = None,
        priority_meta: Optional[PriorityRequestMetadata] = None,
    ):
        """Context manager to select a backend and guarantee release on completion."""
        backend = self.select(client_ip=client_ip, priority_meta=priority_meta)
        success = False
        try:
            yield backend
            success = True
        finally:
            self.release(backend, success=success)


