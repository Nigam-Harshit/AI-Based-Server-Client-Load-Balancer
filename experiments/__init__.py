"""Experiments package for data collection, validation, and execution."""

from .recorder import ExperimentRecord, ExperimentRecorder
from .runner import ExperimentRunner
from .validator import DataValidator
__all__ = [
    "ExperimentRecord",
    "ExperimentRecorder",
    "ExperimentRunner",
    "DataValidator",
    "DatasetAnalyzer",
]


def __getattr__(name: str):
    if name == "DatasetAnalyzer":
        from .analysis import DatasetAnalyzer
        return DatasetAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
