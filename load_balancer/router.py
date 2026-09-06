"""Routing strategies and abstraction for the Load Balancer."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import hashlib
import threading
from typing import Dict, List, Optional


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


ROUTER_REGISTRY = {
    "round_robin": RoundRobinRouter,
    "least_connections": LeastConnectionsRouter,
    "ip_hash": IPHashRouter,
}


def get_router(algorithm: str, backends: List[str]) -> BaseRouter:
    """Factory to instantiate a router by name."""
    normalized = algorithm.lower().replace("-", "_").strip()
    if normalized not in ROUTER_REGISTRY:
        supported = list(ROUTER_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {supported}")
    return ROUTER_REGISTRY[normalized](backends)
