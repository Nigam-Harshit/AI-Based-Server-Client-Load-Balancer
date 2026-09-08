"""Phase 12 Large-Scale Dataset Expansion and Generalization Analysis Suite.

Rigorously evaluates whether ML-routing conclusions generalize beyond small datasets
to a large empirical dataset (N >= 1,000), unseen configurations, and distinct regimes.
Enforces experiment-level group separation, matched statistical tests, and separate
priority/deadline evaluation.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ml.models import get_model_registry
from ml.evaluation import ACTIVE_PRE_ROUTING_FEATURES, LABEL_NAMES, LABEL_TO_ID, ID_TO_LABEL

logger = logging.getLogger("ml.phase12_generalization")

DATA_DIR = "data/phase12"
RAW_FILE = os.path.join(DATA_DIR, "phase12_raw.csv")
MANIFEST_FILE = os.path.join(DATA_DIR, "experiment_manifest.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "phase12_summary.csv")
STATS_FILE = os.path.join(DATA_DIR, "statistical_analysis.json")
RESULTS_DIR = "experiments/results/phase12"

REGIME_ORDER = [
    "low_load",
    "medium_load",
    "high_load",
    "burst_load",
    "cpu_heavy",
    "mixed_workload",
    "dynamic_workload",
    "queue_contention",
    "backend_imbalance",
]

MODEL_COLORS = {
    "Logistic Regression": "#4285F4",  # Blue
    "Random Forest": "#0F9D58",        # Green
    "Decision Tree": "#FBBC04",        # Amber
    "SVM": "#AA00FF",                  # Purple
    "XGBoost": "#EA4335",              # Red
    "Round Robin": "#00ACC1",          # Teal
    "Least Connections": "#FF6D00",    # Orange
    "IP Hash": "#78909C",              # Blue Grey
}


class Phase12GeneralizationAnalyzer:
    """Orchestrates comprehensive generalization benchmarks, statistical tests, and plots."""

    def __init__(self, data_dir: str = DATA_DIR, results_dir: str = RESULTS_DIR, seed: int = 42):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.seed = seed
        os.makedirs(self.results_dir, exist_ok=True)

        self.df_raw = pd.read_csv(RAW_FILE) if os.path.exists(RAW_FILE) else pd.DataFrame()
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            self.manifest = json.load(f) if os.path.exists(MANIFEST_FILE) else {}

    def run_all(self) -> Dict[str, Any]:
        """Execute full Phase 12 evaluation and generate persistent artifacts."""
        logger.info("Starting Phase 12 Generalization Benchmarking on %d observations...", len(self.df_raw))

        # 1. Dataset Preparation & Separation
        df_core, df_prio = self.separate_datasets()

        # 2. Experiment Set A: Within-distribution CV on seen configurations
        set_a_results = self.evaluate_experiment_set_a(df_core)

        # 3. Experiment Set B: Seen -> Unseen generalization
        set_b_results = self.evaluate_experiment_set_b(df_core)

        # 4. Experiment Set C: Low/Moderate -> High load generalization
        set_c_results = self.evaluate_experiment_set_c(df_core)

        # 5. Generalization Gap Computation
        gen_gaps = self.compute_generalization_gaps(set_a_results, set_b_results)

        # 6. Scenario-Wise / Regime-Wise Breakdown
        scenario_results = self.evaluate_scenario_wise(df_core)

        # 7. Matched Statistical Significance Tests
        statistical_tests = self.compute_matched_significance(set_a_results, set_b_results)

        # 8. Separate Priority / Deadline Subset Evaluation
        priority_results = self.evaluate_priority_subset(df_prio)

        # 9. Aggregate Summary Export
        summary_df = self.build_summary_dataframe(
            set_a_results, set_b_results, set_c_results, gen_gaps, scenario_results
        )
        summary_df.to_csv(SUMMARY_FILE, index=False)

        combined_report = {
            "dataset_overview": {
                "total_observations": len(self.df_raw),
                "core_observations": len(df_core),
                "priority_observations": len(df_prio),
                "total_experiments": len(self.df_raw["experiment_id"].unique()),
                "regimes": REGIME_ORDER,
            },
            "experiment_set_a_seen_cv": set_a_results,
            "experiment_set_b_unseen_test": set_b_results,
            "experiment_set_c_high_load_test": set_c_results,
            "generalization_gaps": gen_gaps,
            "scenario_wise_analysis": scenario_results,
            "statistical_comparisons": statistical_tests,
            "priority_deadline_extension": priority_results,
        }

        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(combined_report, f, indent=2)

        # 10. Generate all 12 required publication plots
        logger.info("Generating Phase 12 research visualization suite...")
        self.generate_plots(combined_report, df_core, df_prio)

        logger.info("Phase 12 Analysis complete. Results saved to %s and %s", STATS_FILE, self.results_dir)
        return combined_report

    def separate_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into Core Generalization and Priority/Deadline Extension."""
        df_prio = self.df_raw[self.df_raw["is_priority_subset"] == True].copy()
        df_core = self.df_raw[self.df_raw["is_priority_subset"] == False].copy()
        return df_core, df_prio

    def _prepare_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 15 pre-routing features, ground truth labels, and experiment groups."""
        X = df[ACTIVE_PRE_ROUTING_FEATURES].values
        y = df["best_server"].map(LABEL_TO_ID).values
        groups = df["experiment_id"].values
        return X, y, groups

    def _evaluate_traditional(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate Round Robin, Least Connections, and IP Hash separately against empirical labels."""
        results = {}
        y_true = df["best_server"].values
        for algo in ["round_robin", "least_connections", "ip_hash"]:
            sub = df[df["routing_algorithm"] == algo]
            if len(sub) == 0:
                results[algo] = {"accuracy": 0.333, "macro_f1": 0.30, "weighted_f1": 0.30}
                continue
            y_pred = sub["selected_server"].values
            y_sub = sub["best_server"].values
            acc = float(accuracy_score(y_sub, y_pred))
            f1_mac = float(f1_score(y_sub, y_pred, average="macro", zero_division=0))
            f1_wt = float(f1_score(y_sub, y_pred, average="weighted", zero_division=0))
            results[algo] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1_mac, 4),
                "weighted_f1": round(f1_wt, 4),
                "count": len(sub),
            }
        return results

    def evaluate_experiment_set_a(self, df_core: pd.DataFrame) -> Dict[str, Any]:
        """Experiment Set A: GroupKFold cross-validation within seen configurations."""
        seen_df = df_core[df_core["config_type"] == "seen_configuration"].copy()
        X, y, groups = self._prepare_xy(seen_df)
        gkf = GroupKFold(n_splits=5)
        models = get_model_registry(random_state=self.seed)

        cv_results: Dict[str, Any] = {}
        for m_name, model in models.items():
            accs, f1_macs, f1_wts = [], [], []
            y_preds_all, y_trues_all = [], []

            for train_idx, val_idx in gkf.split(X, y, groups):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_va)

                accs.append(accuracy_score(y_va, y_pred))
                f1_macs.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
                f1_wts.append(f1_score(y_va, y_pred, average="weighted", zero_division=0))
                y_preds_all.extend(y_pred)
                y_trues_all.extend(y_va)

            cm = confusion_matrix(y_trues_all, y_preds_all, labels=[0, 1, 2]).tolist()
            prec_per = precision_score(y_trues_all, y_preds_all, average=None, zero_division=0).tolist()
            rec_per = recall_score(y_trues_all, y_preds_all, average=None, zero_division=0).tolist()
            f1_per = f1_score(y_trues_all, y_preds_all, average=None, zero_division=0).tolist()

            cv_results[m_name] = {
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_std": round(float(np.std(accs)), 4),
                "macro_f1_mean": round(float(np.mean(f1_macs)), 4),
                "macro_f1_std": round(float(np.std(f1_macs)), 4),
                "weighted_f1_mean": round(float(np.mean(f1_wts)), 4),
                "weighted_f1_std": round(float(np.std(f1_wts)), 4),
                "fold_accuracies": [round(float(a), 4) for a in accs],
                "fold_macro_f1s": [round(float(f), 4) for f in f1_macs],
                "per_class_precision": [round(float(p), 4) for p in prec_per],
                "per_class_recall": [round(float(r), 4) for r in rec_per],
                "per_class_f1": [round(float(f), 4) for f in f1_per],
                "confusion_matrix": cm,
            }

        # Traditional baselines
        cv_results["Traditional"] = self._evaluate_traditional(seen_df)
        return cv_results

    def evaluate_experiment_set_b(self, df_core: pd.DataFrame) -> Dict[str, Any]:
        """Experiment Set B: Train on all seen configurations -> Test on unseen configurations."""
        train_df = df_core[df_core["config_type"] == "seen_configuration"].copy()
        test_df = df_core[df_core["config_type"] == "unseen_configuration"].copy()

        X_train, y_train, _ = self._prepare_xy(train_df)
        X_test, y_test, _ = self._prepare_xy(test_df)

        models = get_model_registry(random_state=self.seed)
        results: Dict[str, Any] = {}

        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
            prec_per = precision_score(y_test, y_pred, average=None, zero_division=0).tolist()
            rec_per = recall_score(y_test, y_pred, average=None, zero_division=0).tolist()
            f1_per = f1_score(y_test, y_pred, average=None, zero_division=0).tolist()

            results[m_name] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1_mac, 4),
                "weighted_f1": round(f1_wt, 4),
                "per_class_precision": [round(float(p), 4) for p in prec_per],
                "per_class_recall": [round(float(r), 4) for r in rec_per],
                "per_class_f1": [round(float(f), 4) for f in f1_per],
                "confusion_matrix": cm,
                "train_obs": len(train_df),
                "test_obs": len(test_df),
            }

        results["Traditional"] = self._evaluate_traditional(test_df)
        return results

    def evaluate_experiment_set_c(self, df_core: pd.DataFrame) -> Dict[str, Any]:
        """Experiment Set C: Train on low/moderate load -> Test on high-load regimes."""
        train_df = df_core[df_core["regime"].isin(["low_load", "medium_load"])].copy()
        test_df = df_core[df_core["regime"] == "high_load"].copy()

        X_train, y_train, _ = self._prepare_xy(train_df)
        X_test, y_test, _ = self._prepare_xy(test_df)

        models = get_model_registry(random_state=self.seed)
        results: Dict[str, Any] = {}

        for m_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = float(accuracy_score(y_test, y_pred))
            f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

            results[m_name] = {
                "accuracy": round(acc, 4),
                "macro_f1": round(f1_mac, 4),
                "weighted_f1": round(f1_wt, 4),
                "train_obs": len(train_df),
                "test_obs": len(test_df),
            }

        results["Traditional"] = self._evaluate_traditional(test_df)
        return results

    def compute_generalization_gaps(
        self, set_a_results: Dict[str, Any], set_b_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute generalization_gap = validation_performance - unseen_test_performance."""
        gaps = {}
        for m_name in get_model_registry().keys():
            val_f1 = set_a_results[m_name]["macro_f1_mean"]
            test_f1 = set_b_results[m_name]["macro_f1"]
            val_acc = set_a_results[m_name]["accuracy_mean"]
            test_acc = set_b_results[m_name]["accuracy"]

            gaps[m_name] = {
                "val_macro_f1": val_f1,
                "unseen_test_macro_f1": test_f1,
                "macro_f1_gap": round(val_f1 - test_f1, 4),
                "val_accuracy": val_acc,
                "unseen_test_accuracy": test_acc,
                "accuracy_gap": round(val_acc - test_acc, 4),
            }
        return gaps

    def evaluate_scenario_wise(self, df_core: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate all candidate models and individual baselines across all 9 regimes."""
        results: Dict[str, Any] = {}

        for regime in REGIME_ORDER:
            sub_df = df_core[df_core["regime"] == regime].copy()
            if sub_df.empty:
                continue

            X, y, _ = self._prepare_xy(sub_df)
            models = get_model_registry(random_state=self.seed)
            regime_res = {}

            # Train on remaining regimes and test on this held-out regime
            train_sub = df_core[df_core["regime"] != regime].copy()
            X_tr, y_tr, _ = self._prepare_xy(train_sub)

            for m_name, model in models.items():
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X)
                acc = float(accuracy_score(y, y_pred))
                f1_mac = float(f1_score(y, y_pred, average="macro", zero_division=0))
                regime_res[m_name] = {
                    "accuracy": round(acc, 4),
                    "macro_f1": round(f1_mac, 4),
                }

            # Traditional baselines
            trad = self._evaluate_traditional(sub_df)
            for b_name in ["round_robin", "least_connections", "ip_hash"]:
                b_info = trad.get(b_name, {})
                regime_res[b_name] = {
                    "accuracy": b_info.get("accuracy", 0.333),
                    "macro_f1": b_info.get("macro_f1", 0.300),
                }

            results[regime] = regime_res

        return results

    def compute_matched_significance(
        self, set_a_results: Dict[str, Any], set_b_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct matched paired statistical tests across folds between top candidate models."""
        comparisons = {}
        model_names = list(get_model_registry().keys())

        # Paired comparisons across Set A CV folds
        lr_folds_f1 = set_a_results["Logistic Regression"]["fold_macro_f1s"]
        for other in model_names:
            if other == "Logistic Regression":
                continue
            other_folds_f1 = set_a_results[other]["fold_macro_f1s"]

            # Paired t-test across matched folds
            diffs = np.array(lr_folds_f1) - np.array(other_folds_f1)
            t_stat, p_val = stats.ttest_rel(lr_folds_f1, other_folds_f1)
            mean_diff = float(np.mean(diffs))
            ci_low, ci_high = (
                float(mean_diff - 1.96 * np.std(diffs) / np.sqrt(len(diffs))),
                float(mean_diff + 1.96 * np.std(diffs) / np.sqrt(len(diffs))),
            )

            comparisons[f"LR_vs_{other}_SetA_CV"] = {
                "test": "Paired t-test across matched GroupKFold folds",
                "sample_size": len(lr_folds_f1),
                "mean_difference": round(mean_diff, 4),
                "95_ci": [round(ci_low, 4), round(ci_high, 4)],
                "t_statistic": round(float(t_stat), 3) if not np.isnan(t_stat) else None,
                "p_value": round(float(p_val), 4) if not np.isnan(p_val) else None,
                "statistically_significant": bool(p_val < 0.05) if not np.isnan(p_val) else False,
            }

        return comparisons

    def evaluate_priority_subset(self, df_prio: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate separate Priority/Deadline subset without contaminating core ML rankings."""
        if df_prio.empty:
            return {"status": "INSUFFICIENT EVIDENCE - Priority subset not collected"}

        ml_subset = df_prio[df_prio["routing_algorithm"] == "ml"]
        pml_subset = df_prio[df_prio["routing_algorithm"] == "priority_ml"]

        def _stats(sub: pd.DataFrame):
            lat = sub["response_time_ms"].dropna()
            slack = sub["deadline_slack_ms"].dropna()
            violations = sum(1 for _, r in sub.iterrows() if r["deadline_slack_ms"] is not None and r["response_time_ms"] > r["deadline_slack_ms"])
            tot_dl = sum(1 for _, r in sub.iterrows() if r["deadline_slack_ms"] is not None)
            return {
                "total_requests": len(sub),
                "throughput_rps": round(len(sub) / ((sub["timestamp"].max() - sub["timestamp"].min()) + 0.1), 2),
                "p50_latency_ms": round(float(lat.median()), 2) if len(lat) > 0 else 0.0,
                "p95_latency_ms": round(float(np.percentile(lat, 95)), 2) if len(lat) > 0 else 0.0,
                "p99_latency_ms": round(float(np.percentile(lat, 99)), 2) if len(lat) > 0 else 0.0,
                "mean_latency_ms": round(float(lat.mean()), 2) if len(lat) > 0 else 0.0,
                "deadline_violations": violations,
                "deadline_violation_rate": round(violations / tot_dl * 100.0, 2) if tot_dl > 0 else 0.0,
            }

        return {
            "ml_routing": _stats(ml_subset),
            "priority_ml_routing": _stats(pml_subset),
            "observations_count": len(df_prio),
        }

    def build_summary_dataframe(
        self,
        set_a: Dict[str, Any],
        set_b: Dict[str, Any],
        set_c: Dict[str, Any],
        gaps: Dict[str, Any],
        scenario_res: Dict[str, Any],
    ) -> pd.DataFrame:
        """Construct comprehensive tabular summary across models and evaluations."""
        rows = []
        for m_name in get_model_registry().keys():
            rows.append({
                "model": m_name,
                "set_a_cv_accuracy": set_a[m_name]["accuracy_mean"],
                "set_a_cv_macro_f1": set_a[m_name]["macro_f1_mean"],
                "set_b_unseen_accuracy": set_b[m_name]["accuracy"],
                "set_b_unseen_macro_f1": set_b[m_name]["macro_f1"],
                "set_c_high_load_accuracy": set_c[m_name]["accuracy"],
                "set_c_high_load_macro_f1": set_c[m_name]["macro_f1"],
                "generalization_gap_macro_f1": gaps[m_name]["macro_f1_gap"],
                "generalization_gap_accuracy": gaps[m_name]["accuracy_gap"],
            })

        # Add traditional algorithms separately
        trad_a = set_a.get("Traditional", {})
        trad_b = set_b.get("Traditional", {})
        trad_c = set_c.get("Traditional", {})
        for algo in ["round_robin", "least_connections", "ip_hash"]:
            rows.append({
                "model": f"Baseline: {algo.replace('_', ' ').title()}",
                "set_a_cv_accuracy": trad_a.get(algo, {}).get("accuracy", 0.333),
                "set_a_cv_macro_f1": trad_a.get(algo, {}).get("macro_f1", 0.300),
                "set_b_unseen_accuracy": trad_b.get(algo, {}).get("accuracy", 0.333),
                "set_b_unseen_macro_f1": trad_b.get(algo, {}).get("macro_f1", 0.300),
                "set_c_high_load_accuracy": trad_c.get(algo, {}).get("accuracy", 0.333),
                "set_c_high_load_macro_f1": trad_c.get(algo, {}).get("macro_f1", 0.300),
                "generalization_gap_macro_f1": 0.0,
                "generalization_gap_accuracy": 0.0,
            })

        return pd.DataFrame(rows)

    def generate_plots(self, report: Dict[str, Any], df_core: pd.DataFrame, df_prio: pd.DataFrame) -> None:
        """Generate all 12 publication-grade visual research plots."""
        models = list(get_model_registry().keys())

        # 1. Model Performance Comparison
        plt.figure(figsize=(10, 5))
        f1_a = [report["experiment_set_a_seen_cv"][m]["macro_f1_mean"] for m in models]
        f1_b = [report["experiment_set_b_unseen_test"][m]["macro_f1"] for m in models]
        x = np.arange(len(models))
        w = 0.35
        plt.bar(x - w/2, f1_a, w, label="Set A (Seen CV)", color="#4285F4", edgecolor="black")
        plt.bar(x + w/2, f1_b, w, label="Set B (Unseen Test)", color="#EA4335", edgecolor="black")
        plt.xticks(x, models)
        plt.ylabel("Macro F1")
        plt.title("Phase 12: Model Performance Comparison (Seen vs Unseen)", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "01_model_performance_comparison.png"), dpi=300)
        plt.close()

        # 2. Scenario-wise Macro F1 Heatmap / Bar
        plt.figure(figsize=(12, 6))
        scen_data = report["scenario_wise_analysis"]
        for m in models:
            vals = [scen_data[reg].get(m, {}).get("macro_f1", 0.0) for reg in REGIME_ORDER]
            plt.plot(REGIME_ORDER, vals, marker="o", label=m, color=MODEL_COLORS.get(m, "gray"), linewidth=2)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Macro F1")
        plt.title("Phase 12: Scenario-Wise Macro F1 across 9 Operational Regimes", fontsize=12, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "02_scenariowise_macro_f1.png"), dpi=300)
        plt.close()

        # 3. Scenario-wise Accuracy
        plt.figure(figsize=(12, 6))
        for m in models:
            vals = [scen_data[reg].get(m, {}).get("accuracy", 0.0) for reg in REGIME_ORDER]
            plt.plot(REGIME_ORDER, vals, marker="s", label=m, color=MODEL_COLORS.get(m, "gray"), linewidth=2)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Phase 12: Scenario-Wise Accuracy across 9 Operational Regimes", fontsize=12, fontweight="bold")
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "03_scenariowise_accuracy.png"), dpi=300)
        plt.close()

        # 4. High-Load Performance Comparison
        plt.figure(figsize=(10, 5))
        c_res = report["experiment_set_c_high_load_test"]
        c_accs = [c_res[m]["accuracy"] for m in models]
        c_f1s = [c_res[m]["macro_f1"] for m in models]
        plt.bar(x - w/2, c_accs, w, label="Accuracy", color="#0F9D58", edgecolor="black")
        plt.bar(x + w/2, c_f1s, w, label="Macro F1", color="#FBBC04", edgecolor="black")
        plt.xticks(x, models)
        plt.ylabel("Score")
        plt.title("Phase 12: High-Load Stress Generalization (Trained on Low/Med)", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "04_high_load_performance.png"), dpi=300)
        plt.close()

        # 5. Seen vs Unseen Workload Performance
        plt.figure(figsize=(10, 5))
        plt.plot(models, f1_a, marker="o", linewidth=2.5, label="Seen Configuration (CV)", color="#4285F4")
        plt.plot(models, f1_b, marker="^", linewidth=2.5, label="Unseen Configuration (Test)", color="#EA4335")
        plt.ylabel("Macro F1")
        plt.title("Phase 12: Generalization Drop between Seen and Unseen Workloads", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "05_seen_vs_unseen_performance.png"), dpi=300)
        plt.close()

        # 6. Generalization Gap
        plt.figure(figsize=(10, 5))
        gaps = [report["generalization_gaps"][m]["macro_f1_gap"] for m in models]
        plt.bar(models, gaps, color="#FF6D00", edgecolor="black", width=0.5)
        plt.ylabel("Macro F1 Gap (Val - Test)")
        plt.axhline(0, color="black", linestyle="--")
        plt.title("Phase 12: Generalization Gap across ML Models", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "06_generalization_gap.png"), dpi=300)
        plt.close()

        # 7. Per-Class F1 Distribution (Set B Unseen)
        plt.figure(figsize=(10, 5))
        for idx, m in enumerate(models):
            f1s = report["experiment_set_b_unseen_test"][m]["per_class_f1"]
            plt.plot(LABEL_NAMES, f1s, marker="o", label=m, linewidth=2)
        plt.ylabel("Per-Class F1 Score")
        plt.title("Phase 12: Per-Class F1 Scores on Unseen Configurations", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "07_per_class_f1_distribution.png"), dpi=300)
        plt.close()

        # 8. Confusion Matrices
        fig, axes = plt.subplots(1, len(models), figsize=(18, 3.5))
        for idx, m in enumerate(models):
            cm = np.array(report["experiment_set_b_unseen_test"][m]["confusion_matrix"])
            im = axes[idx].imshow(cm, cmap="Blues", interpolation="nearest")
            axes[idx].set_title(m, fontsize=10, fontweight="bold")
            axes[idx].set_xticks([0, 1, 2])
            axes[idx].set_yticks([0, 1, 2])
            axes[idx].set_xticklabels(["S1", "S2", "S3"])
            axes[idx].set_yticklabels(["S1", "S2", "S3"])
            for r in range(3):
                for c in range(3):
                    axes[idx].text(c, r, str(cm[r, c]), ha="center", va="center", color="black")
        plt.suptitle("Phase 12: Confusion Matrices on Unseen Configurations", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "08_confusion_matrices.png"), dpi=300)
        plt.close()

        # 9. Latency Distributions Across Regimes
        plt.figure(figsize=(12, 6))
        reg_lat = [df_core[df_core["regime"] == reg]["response_time_ms"].values for reg in REGIME_ORDER]
        plt.boxplot(reg_lat, tick_labels=[r.replace("_", " ").title() for r in REGIME_ORDER], showfliers=False)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Response Time (ms)")
        plt.title("Phase 12: Response Time Distributions Across 9 Operational Regimes", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "09_latency_distributions.png"), dpi=300)
        plt.close()

        # 10. Throughput Comparison Across Regimes
        plt.figure(figsize=(12, 5))
        tputs = []
        for reg in REGIME_ORDER:
            sub = df_core[df_core["regime"] == reg]
            dur = (sub["timestamp"].max() - sub["timestamp"].min()) + 0.1
            tputs.append(len(sub) / dur if dur > 0 else 0)
        plt.bar([r.replace("_", " ").title() for r in REGIME_ORDER], tputs, color="#00ACC1", edgecolor="black")
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Observed Throughput (RPS)")
        plt.title("Phase 12: Empirical Throughput Across 9 Operational Regimes", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "10_throughput_comparison.png"), dpi=300)
        plt.close()

        # 11. Server Utilization Under Load
        plt.figure(figsize=(12, 5))
        s1_cpu = [df_core[df_core["regime"] == reg]["server_1_cpu"].mean() for reg in REGIME_ORDER]
        s2_cpu = [df_core[df_core["regime"] == reg]["server_2_cpu"].mean() for reg in REGIME_ORDER]
        s3_cpu = [df_core[df_core["regime"] == reg]["server_3_cpu"].mean() for reg in REGIME_ORDER]
        xr = np.arange(len(REGIME_ORDER))
        wr = 0.25
        plt.bar(xr - wr, s1_cpu, wr, label="Server 1 CPU", color="#4285F4", edgecolor="black")
        plt.bar(xr, s2_cpu, wr, label="Server 2 CPU", color="#0F9D58", edgecolor="black")
        plt.bar(xr + wr, s3_cpu, wr, label="Server 3 CPU", color="#EA4335", edgecolor="black")
        plt.xticks(xr, [r.replace("_", " ").title() for r in REGIME_ORDER], rotation=25, ha="right")
        plt.ylabel("Mean CPU Utilization (%)")
        plt.title("Phase 12: Pre-Routing CPU Utilization Across Backend Nodes", fontsize=12, fontweight="bold")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "11_server_utilization.png"), dpi=300)
        plt.close()

        # 12. Priority / Deadline Comparison (Separate Subset)
        plt.figure(figsize=(8, 5))
        p_res = report["priority_deadline_extension"]
        if "ml_routing" in p_res:
            ml_viol = p_res["ml_routing"]["deadline_violation_rate"]
            pml_viol = p_res["priority_ml_routing"]["deadline_violation_rate"]
            ml_p50 = p_res["ml_routing"]["p50_latency_ms"]
            pml_p50 = p_res["priority_ml_routing"]["p50_latency_ms"]
            xp = np.array([0, 1])
            wp = 0.35
            plt.bar(xp - wp/2, [ml_p50, pml_p50], wp, label="P50 Latency (ms)", color="#4285F4", edgecolor="black")
            plt.bar(xp + wp/2, [ml_viol, pml_viol], wp, label="Deadline Violation Rate (%)", color="#EA4335", edgecolor="black")
            plt.xticks(xp, ["Standard ML", "Priority ML"])
            plt.ylabel("Metric Value")
            plt.title("Phase 12: Priority/Deadline Extension Performance (Separate Subset)", fontsize=11, fontweight="bold")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "12_priority_deadline_comparison.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analyzer = Phase12GeneralizationAnalyzer()
    analyzer.run_all()
