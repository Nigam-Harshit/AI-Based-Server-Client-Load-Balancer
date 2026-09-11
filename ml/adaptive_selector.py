"""Adaptive / Context-Aware Model Selection for ML Load Balancing.

Phase 13 Core Component:
Implements two adaptive model-selection strategies:
1. Strategy A: Evidence-Based Policy Selector (systematically maps operating regime to best historical model).
2. Strategy B: Learned Meta-Selector (lightweight classifier predicting optimal model from pre-routing context).

Includes robust fallback to LeastConnectionsRouter, zero-leakage enforcement,
and model-switching tracking.
"""

import enum
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from config.backends import DEFAULT_BACKENDS
from load_balancer.router import BaseRouter, LeastConnectionsRouter, MLRouter
from ml.evaluation import ACTIVE_PRE_ROUTING_FEATURES, LABEL_TO_ID, ID_TO_LABEL
from monitoring.collector import ServerMetrics

logger = logging.getLogger("ml.adaptive_selector")

CANDIDATE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "SVM",
    "XGBoost",
]

MODEL_KEY_MAP = {
    "logistic_regression": "Logistic Regression",
    "logisticregression": "Logistic Regression",
    "lr": "Logistic Regression",
    "random_forest": "Random Forest",
    "randomforest": "Random Forest",
    "rf": "Random Forest",
    "decision_tree": "Decision Tree",
    "decisiontree": "Decision Tree",
    "dt": "Decision Tree",
    "svm": "SVM",
    "svc": "SVM",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
}

DEFAULT_MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Random Forest": "models/random_forest.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "SVM": "models/svm.joblib",
    "XGBoost": "models/xgboost.joblib",
}


class AdaptiveStrategy(enum.Enum):
    """Adaptive model selection strategies."""
    POLICY = "policy"
    META = "meta"

    @classmethod
    def from_string(cls, val: str) -> "AdaptiveStrategy":
        normalized = str(val).lower().strip()
        if normalized in ("meta", "learned", "learned_meta"):
            return cls.META
        return cls.POLICY


class EvidenceBasedPolicySelector:
    """Strategy A: Evidence-Based Policy Selector.

    Identifies operating regime from pre-routing telemetry and assigns the model
    that systematically performed best on training/cross-validation experiments.
    Never uses held-out test data for policy rules.
    """

    def __init__(self, policy_table: Optional[Dict[str, str]] = None):
        """Initialize with evidence-based policy lookup table.

        Default policy is derived from Phase 12 training/CV evidence:
        - Symmetric / Balanced traffic (low, medium, dynamic): SVM (highest CV Macro F1: 92.39%).
        - High Load / Stress / Queue contention: Random Forest / Tree-Ensemble (high robustness, 100% transfer).
        - Backend Imbalance (high variance between nodes): SVM (highest resilience under node degradation: 0.1607 Macro F1).
        """
        self.policy_table = policy_table or {
            "low_load": "SVM",
            "medium_load": "SVM",
            "high_load": "Random Forest",
            "burst_load": "Random Forest",
            "cpu_heavy": "Random Forest",
            "mixed_workload": "SVM",
            "dynamic_workload": "SVM",
            "queue_contention": "Random Forest",
            "backend_imbalance": "SVM",
            "default": "SVM",
        }

    def infer_regime(
        self,
        df_features: pd.DataFrame,
        metrics_by_backend: Optional[Dict[str, ServerMetrics]] = None,
    ) -> str:
        """Derive the operational regime strictly from pre-routing features."""
        if df_features is None or df_features.empty:
            return "default"

        row = df_features.iloc[0]

        # Extract CPU values across all 3 nodes
        cpus = [
            float(row.get("server_1_cpu", 0.0)),
            float(row.get("server_2_cpu", 0.0)),
            float(row.get("server_3_cpu", 0.0)),
        ]
        conns = [
            float(row.get("server_1_connections", 0.0)),
            float(row.get("server_2_connections", 0.0)),
            float(row.get("server_3_connections", 0.0)),
        ]
        resps = [
            float(row.get("server_1_response_time", 0.0)),
            float(row.get("server_2_response_time", 0.0)),
            float(row.get("server_3_response_time", 0.0)),
        ]

        max_cpu = max(cpus)
        min_cpu = min(cpus)
        cpu_spread = max_cpu - min_cpu
        mean_cpu = np.mean(cpus)
        total_conns = sum(conns)
        mean_resp = np.mean(resps)

        # Asymmetric node stress / Backend Imbalance
        if cpu_spread > 40.0 or (max_cpu > 70.0 and min_cpu < 30.0):
            return "backend_imbalance"

        # Concurrency & Queue saturation / Queue Contention
        if total_conns >= 8.0:
            return "queue_contention"

        # Heavy compute saturation
        if mean_cpu > 65.0:
            return "cpu_heavy"

        # High overall load
        if total_conns >= 5.0 or mean_resp > 50.0:
            return "high_load"

        # Medium load
        if total_conns >= 3.0 or mean_resp > 25.0:
            return "medium_load"

        # Low load
        return "low_load"

    def select_model(
        self,
        df_features: pd.DataFrame,
        metrics_by_backend: Optional[Dict[str, ServerMetrics]] = None,
    ) -> Tuple[str, float, str]:
        """Select model using evidence-based regime mapping.

        Returns:
            (selected_model_name, selector_confidence, regime)
        """
        regime = self.infer_regime(df_features, metrics_by_backend)
        selected = self.policy_table.get(regime, self.policy_table.get("default", "SVM"))
        # High confidence for deterministic policy mapping
        confidence = 0.95
        return selected, confidence, regime


class LearnedMetaSelector:
    """Strategy B: Learned Meta-Selector.

    A lightweight machine-learning classifier (DecisionTree or LogisticRegression)
    trained on pre-routing context vectors to predict which candidate model
    achieves the highest routing performance.
    """

    def __init__(
        self,
        meta_model: Optional[Any] = None,
        candidate_models: Optional[List[str]] = None,
        confidence_threshold: float = 0.40,
    ):
        self.candidate_models = candidate_models or list(CANDIDATE_MODELS)
        self.confidence_threshold = confidence_threshold
        self.meta_model = meta_model or DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=5,
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, X_context: pd.DataFrame, y_best_model: pd.Series) -> "LearnedMetaSelector":
        """Fit meta-selector strictly on training experiments."""
        # Convert model names to categorical codes
        self.classes_ = np.array(sorted(list(set(y_best_model.unique()))))
        self.meta_model.fit(X_context, y_best_model)
        self.is_fitted = True
        return self

    def select_model(self, df_features: pd.DataFrame) -> Tuple[str, float, str]:
        """Predict optimal model from pre-routing context features.

        Returns:
            (selected_model_name, confidence, reason)
        """
        if not self.is_fitted:
            # Unfitted fallback: return default robust model (SVM)
            return "SVM", 0.50, "meta_unfitted_default"

        try:
            pred = self.meta_model.predict(df_features)[0]
            confidence = 0.50
            if hasattr(self.meta_model, "predict_proba"):
                probs = self.meta_model.predict_proba(df_features)[0]
                confidence = float(np.max(probs))

            selected = str(pred)
            if selected not in self.candidate_models:
                selected = "SVM"

            reason = f"meta_predicted_conf_{confidence:.2f}"
            return selected, confidence, reason
        except Exception as e:
            logger.warning("Meta-selector prediction failed: %s; falling back to SVM", e)
            return "SVM", 0.30, f"meta_exception_{str(e)}"


class AdaptiveRouter(BaseRouter):
    """Adaptive / Context-Aware Router integrating candidate models and adaptive selection."""

    def __init__(
        self,
        backends: List[str],
        collector: Optional[Any] = None,
        strategy: str = "policy",
        model_paths: Optional[Dict[str, str]] = None,
        confidence_threshold: float = 0.35,
    ):
        super().__init__(backends)
        self.collector = collector
        self.strategy = AdaptiveStrategy.from_string(strategy)
        self.confidence_threshold = confidence_threshold

        # Fallback router: LeastConnectionsRouter
        self.fallback_router = LeastConnectionsRouter(backends=self.backends)

        # Strategy selectors
        self.policy_selector = EvidenceBasedPolicySelector()
        self.meta_selector = LearnedMetaSelector()
        if os.path.exists("models/meta_selector.joblib"):
            try:
                self.meta_selector.meta_model = joblib.load("models/meta_selector.joblib")
                self.meta_selector.is_fitted = True
            except Exception:
                pass

        # Load candidate models
        self.model_paths = model_paths or dict(DEFAULT_MODEL_PATHS)
        self.models: Dict[str, Any] = {}
        self._load_candidate_models()

        # State tracking
        self.current_model: Optional[str] = None
        self.switch_count: int = 0
        self.selection_counts: Dict[str, int] = {m: 0 for m in CANDIDATE_MODELS}
        self.total_routed_requests: int = 0
        self.fallback_count: int = 0

        # Last adaptive decision metadata
        self.last_adaptive_decision: Dict[str, Any] = {}
        self.last_prediction: Dict[str, Any] = {}

    def _load_candidate_models(self) -> None:
        """Load candidate model artifacts from disk with graceful fallbacks."""
        for model_name, path in self.model_paths.items():
            if os.path.exists(path):
                try:
                    artifact = joblib.load(path)
                    self.models[model_name] = artifact
                    logger.info("Loaded candidate model %s from %s", model_name, path)
                except Exception as e:
                    logger.warning("Failed to load model %s from %s: %s", model_name, path, e)
            else:
                logger.warning("Model artifact path %s for %s does not exist", path, model_name)

    def set_healthy_backends(self, healthy: Optional[List[str]]) -> None:
        super().set_healthy_backends(healthy)
        if hasattr(self.fallback_router, "set_healthy_backends"):
            self.fallback_router.set_healthy_backends(healthy)

    def select(
        self,
        client_ip: Optional[str] = None,
        priority_meta: Optional[Any] = None,
    ) -> str:
        """Execute adaptive routing decision."""
        self.total_routed_requests += 1
        t_start = time.perf_counter()

        # 1. Collector health check
        if self.collector is None:
            return self._fallback_routing(
                client_ip=client_ip,
                reason="Collector unavailable",
                selection_time_ms=0.0,
            )

        try:
            # 2. Collect pre-routing metrics
            metrics_by_backend = self.collector.collect_all()

            # 3. Extract pre-routing features (Strict zero-leakage contract)
            from ml.features import extract_features_from_metrics
            df_features, availability_map = extract_features_from_metrics(
                metrics_by_backend, backends=self.backends
            )

            # Check healthy backends
            healthy_backends = [b for b in self.backends if availability_map.get(b, True)]
            if getattr(self, "_healthy_backends", None) is not None:
                healthy_backends = [b for b in healthy_backends if b in self._healthy_backends]
            if not healthy_backends:
                t_sel = round((time.perf_counter() - t_start) * 1000.0, 3)
                return self._fallback_routing(
                    client_ip=client_ip,
                    reason="No backends healthy",
                    selection_time_ms=t_sel,
                )

            # 4. Adaptive Model Selection
            t_sel_start = time.perf_counter()
            if self.strategy == AdaptiveStrategy.META and self.meta_selector.is_fitted:
                selected_model, selector_conf, selection_reason = self.meta_selector.select_model(df_features)
            else:
                selected_model, selector_conf, selection_reason = self.policy_selector.select_model(
                    df_features, metrics_by_backend
                )
            selection_time_ms = round((time.perf_counter() - t_sel_start) * 1000.0, 3)

            # Track model switching
            if self.current_model is not None and selected_model != self.current_model:
                self.switch_count += 1
            self.current_model = selected_model
            self.selection_counts[selected_model] = self.selection_counts.get(selected_model, 0) + 1

            # 5. Model Availability & Inference
            model = self.models.get(selected_model)
            if model is None:
                return self._fallback_routing(
                    client_ip=client_ip,
                    reason=f"Selected model '{selected_model}' artifact not loaded",
                    selection_time_ms=selection_time_ms,
                    selected_model=selected_model,
                )

            # Model inference
            t_inf_start = time.perf_counter()
            raw_pred = model.predict(df_features)[0]
            if isinstance(raw_pred, (int, np.integer)):
                pred_label = ID_TO_LABEL.get(int(raw_pred), f"server-{raw_pred + 1}")
            else:
                pred_label = str(raw_pred)

            ml_conf = 1.0
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(df_features)[0]
                    if isinstance(raw_pred, (int, np.integer)) and int(raw_pred) < len(probs):
                        ml_conf = float(probs[int(raw_pred)])
                    else:
                        ml_conf = float(np.max(probs))
                except Exception:
                    ml_conf = 1.0

            inference_time_ms = round((time.perf_counter() - t_inf_start) * 1000.0, 3)

            # Label to backend URL mapping
            label_to_backend = {
                f"server-{i+1}": b for i, b in enumerate(self.backends)
            }
            predicted_backend = label_to_backend.get(pred_label)

            # 6. Safety Validation (Backend health & confidence)
            if (
                predicted_backend
                and predicted_backend in healthy_backends
                and ml_conf >= self.confidence_threshold
            ):
                chosen = predicted_backend
                is_fallback = False
                fallback_reason = None
            else:
                chosen = self.fallback_router.select(client_ip=client_ip)
                is_fallback = True
                self.fallback_count += 1
                fallback_reason = (
                    f"Low confidence ({ml_conf:.2f} < {self.confidence_threshold:.2f})"
                    if ml_conf < self.confidence_threshold
                    else f"Predicted backend {pred_label} unavailable"
                )

            # Record decision metadata
            switch_rate = round(self.switch_count / max(1, self.total_routed_requests), 4)
            p_name = None
            slack_ms = None
            if priority_meta is not None:
                p_level = getattr(priority_meta, "priority", None)
                p_name = getattr(p_level, "name", str(p_level)) if p_level else None
                if hasattr(priority_meta, "calculate_slack_ms"):
                    slack_ms = priority_meta.calculate_slack_ms()

            self.last_adaptive_decision = {
                "selected_model": selected_model,
                "strategy": self.strategy.value,
                "selection_reason": selection_reason,
                "selector_confidence": round(selector_conf, 4),
                "model_selection_time_ms": selection_time_ms,
                "ml_predicted_server": pred_label,
                "ml_confidence": round(ml_conf, 4),
                "is_fallback": is_fallback,
                "fallback_reason": fallback_reason,
                "chosen_server": chosen,
                "switch_count": self.switch_count,
                "switch_rate": switch_rate,
                "inference_time_ms": inference_time_ms,
                "priority": p_name,
                "deadline_slack_ms": slack_ms,
            }
            # For backward compatibility with standard ML response headers
            self.last_prediction = {
                "predicted_server": predicted_backend or pred_label,
                "confidence": round(ml_conf, 4),
                "is_fallback": is_fallback,
                "fallback_reason": fallback_reason,
                "chosen_server": chosen,
                "inference_latency_ms": inference_time_ms,
            }
            return chosen

        except Exception as e:
            t_sel = round((time.perf_counter() - t_start) * 1000.0, 3)
            return self._fallback_routing(
                client_ip=client_ip,
                reason=f"Adaptive selection exception: {str(e)}",
                selection_time_ms=t_sel,
            )

    def _fallback_routing(
        self,
        client_ip: Optional[str],
        reason: str,
        selection_time_ms: float,
        selected_model: str = "Fallback-LC",
    ) -> str:
        """Route to LeastConnections fallback when selection or inference fails."""
        self.fallback_count += 1
        chosen = self.fallback_router.select(client_ip=client_ip)
        switch_rate = round(self.switch_count / max(1, self.total_routed_requests), 4)

        self.last_adaptive_decision = {
            "selected_model": selected_model,
            "strategy": self.strategy.value,
            "selection_reason": "fallback",
            "selector_confidence": 0.0,
            "model_selection_time_ms": selection_time_ms,
            "ml_predicted_server": None,
            "ml_confidence": None,
            "is_fallback": True,
            "fallback_reason": reason,
            "chosen_server": chosen,
            "switch_count": self.switch_count,
            "switch_rate": switch_rate,
            "inference_time_ms": 0.0,
        }
        self.last_prediction = {
            "predicted_server": None,
            "confidence": None,
            "is_fallback": True,
            "fallback_reason": reason,
            "chosen_server": chosen,
            "inference_latency_ms": selection_time_ms,
        }
        return chosen

    def release(self, backend: str, success: bool = True) -> None:
        """Release backend on fallback router (which maintains connection counts)."""
        if hasattr(self.fallback_router, "release"):
            self.fallback_router.release(backend, success=success)

    def get_selection_stats(self) -> Dict[str, Any]:
        """Return cumulative summary of model selections, switches, and fallbacks."""
        return {
            "total_routed_requests": self.total_routed_requests,
            "switch_count": self.switch_count,
            "switch_rate": round(self.switch_count / max(1, self.total_routed_requests), 4),
            "fallback_count": self.fallback_count,
            "fallback_rate": round(self.fallback_count / max(1, self.total_routed_requests), 4),
            "selection_counts": dict(self.selection_counts),
        }
