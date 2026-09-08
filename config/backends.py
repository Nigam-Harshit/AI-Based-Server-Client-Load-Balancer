"""Default backend server configuration."""
"""Default backend server configuration with dynamic environment variable support."""

DEFAULT_BACKENDS = [
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
]
import os
from typing import List


def get_configured_backends() -> List[str]:
    """Retrieve configured backend URLs from environment or return defaults."""
    env_backends = os.getenv("BACKENDS") or os.getenv("BACKEND_URLS")
    if env_backends:
        parsed = [b.strip() for b in env_backends.split(",") if b.strip()]
        if parsed:
            return parsed
    return [
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
        "http://127.0.0.1:8003",
    ]


DEFAULT_BACKENDS = get_configured_backends()

