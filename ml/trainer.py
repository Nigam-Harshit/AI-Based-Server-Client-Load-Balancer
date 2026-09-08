"""Production model training and artifact serialization for ML Load Balancer."""

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .evaluation import (
    ModelEvaluator,
    ACTIVE_PRE_ROUTING_FEATURES,
    LABEL_NAMES,
    LABEL_TO_ID,
    ID_TO_LABEL,
)

logger = logging.getLogger("ml.trainer")

DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "logistic_regression.joblib")
DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, "metadata.json")


def train_and_persist_model(
    model_type: str = "LogisticRegression",
    data_dir: str = "data/processed",
    output_dir: str = DEFAULT_MODEL_DIR,
    random_state: int = 42,
) -> Tuple[str, str]:
    """Train validated model pipeline on experimental dataset and persist artifact + metadata."""
    os.makedirs(output_dir, exist_ok=True)
    evaluator = ModelEvaluator(random_state=random_state)
    df = evaluator.load_dataset(data_dir=data_dir)
    X, y, _ = evaluator.prepare_data(df)

    if model_type.lower() in ("logistic_regression", "logisticregression", "lr"):
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
            )),
        ])
        artifact_name = "logistic_regression.joblib"
    elif model_type.lower() in ("random_forest", "randomforest", "rf"):
        pipeline = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=4,
                random_state=random_state,
            )),
        ])
        artifact_name = "random_forest.joblib"
    elif model_type.lower() in ("decision_tree", "decisiontree", "dt"):
        from sklearn.tree import DecisionTreeClassifier
        pipeline = Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=4,
                random_state=random_state,
            )),
        ])
        artifact_name = "decision_tree.joblib"
    elif model_type.lower() in ("svm", "svc"):
        from sklearn.svm import SVC
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=random_state,
            )),
        ])
        artifact_name = "svm.joblib"
    elif model_type.lower() in ("xgboost", "xgb"):
        import xgboost as xgb
        pipeline = Pipeline([
            ("clf", xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric="mlogloss",
            )),
        ])
        artifact_name = "xgboost.joblib"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info("Training %s on %d experimental samples...", model_type, len(df))
    pipeline.fit(X, y)

    artifact_path = os.path.join(output_dir, artifact_name)
    joblib.dump(pipeline, artifact_path)
    logger.info("Saved serialized model pipeline to %s", artifact_path)

    metadata = {
        "model_type": model_type,
        "artifact_path": artifact_path,
        "feature_names": ACTIVE_PRE_ROUTING_FEATURES,
        "feature_count": len(ACTIVE_PRE_ROUTING_FEATURES),
        "target_labels": LABEL_NAMES,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "training_samples": len(df),
        "random_seed": random_state,
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", metadata_path)

    return artifact_path, metadata_path


def train_all_candidate_models(
    data_dir: str = "data/processed",
    output_dir: str = DEFAULT_MODEL_DIR,
    random_state: int = 42,
) -> Dict[str, str]:
    """Train and persist all 5 candidate models for adaptive load balancing."""
    models_to_train = ["LogisticRegression", "RandomForest", "DecisionTree", "SVM", "XGBoost"]
    paths = {}
    for m in models_to_train:
        art_path, _ = train_and_persist_model(
            model_type=m,
            data_dir=data_dir,
            output_dir=output_dir,
            random_state=random_state,
        )
        paths[m] = art_path
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_and_persist_model()
