"""Model definitions and preprocessing pipelines for load balancer classification."""

from typing import Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_model_registry(random_state: int = 42) -> Dict[str, Any]:
    """Return dictionary of models and pipelines for comparison.

    Models requiring feature normalization (Logistic Regression, SVM) include
    a StandardScaler in their Pipeline so preprocessing is fit only on training folds.
    Tree-based models (Random Forest, Decision Tree, XGBoost) operate directly
    on raw continuous values without distortion.
    """
    registry = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            random_state=random_state,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state,
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=random_state,
            )),
        ]),
    }

    if HAS_XGBOOST:
        registry["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
        )

    return registry
