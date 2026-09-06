"""Traditional load balancing baseline evaluation against optimal targets."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SERVER_URL_MAP = {
    "http://127.0.0.1:8001": "server-1",
    "http://127.0.0.1:8002": "server-2",
    "http://127.0.0.1:8003": "server-3",
    "http://localhost:8001": "server-1",
    "http://localhost:8002": "server-2",
    "http://localhost:8003": "server-3",
}


def evaluate_traditional_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the conventional routing algorithm selections against empirical best_server.

    Maps selected_server URL to server ID (e.g. server-1) and evaluates
    classification metrics against best_server ground truth on the identical observations.
    """
    if df.empty or "selected_server" not in df.columns or "best_server" not in df.columns:
        raise ValueError("DataFrame must contain 'selected_server' and 'best_server' columns")

    # Map URLs to standard IDs
    y_true = df["best_server"].astype(str)
    y_pred = df["selected_server"].map(
        lambda s: SERVER_URL_MAP.get(str(s).rstrip("/"), str(s))
    )

    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))

    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_prec = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "model_name": "Traditional Baseline",
        "labels": labels,
        "accuracy": float(acc),
        "macro_precision": float(macro_prec),
        "weighted_precision": float(weighted_prec),
        "macro_recall": float(macro_rec),
        "weighted_recall": float(weighted_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(df),
    }
