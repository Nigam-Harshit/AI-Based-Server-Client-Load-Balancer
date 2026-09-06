"""Machine learning baselines, model comparison, and evaluation package."""

from .baseline import evaluate_traditional_baseline
from .models import get_model_registry
__all__ = [
    "evaluate_traditional_baseline",
    "get_model_registry",
    "ModelEvaluator",
    "ACTIVE_PRE_ROUTING_FEATURES",
    "ZERO_VARIANCE_FEATURES",
    "STRICTLY_EXCLUDED_COLUMNS",
    "LABEL_NAMES",
]


def __getattr__(name: str):
    if name in [
        "ModelEvaluator",
        "ACTIVE_PRE_ROUTING_FEATURES",
        "ZERO_VARIANCE_FEATURES",
        "STRICTLY_EXCLUDED_COLUMNS",
        "LABEL_NAMES",
    ]:
        from . import evaluation
        return getattr(evaluation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
