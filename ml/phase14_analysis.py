"""Phase 14: End-to-End System Benchmark Analysis and Visualization Suite.

Generates:
1. data/phase14/phase14_summary.csv (Aggregated performance, reliability, routing quality, and overhead metrics)
2. data/phase14/statistical_analysis.json (Matched paired hypothesis tests, 95% CIs, Cohen's d effect sizes)
3. 15 Publication-quality figures in experiments/results/phase14/
4. Numerical TABLES A - E for docs/phase14_system_benchmark.md
"""

import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("ml.phase14_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "phase14")
OUTPUT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "phase14")
RAW_CSV_PATH = os.path.join(OUTPUT_DATA_DIR, "phase14_raw.csv")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DATA_DIR, "phase14_summary.csv")
STATS_JSON_PATH = os.path.join(OUTPUT_DATA_DIR, "statistical_analysis.json")

ALGORITHMS = [
    "round_robin",
    "least_connections",
    "ip_hash",
    "logistic_regression",
    "random_forest",
    "decision_tree",
    "svm",
    "xgboost",
    "adaptive_policy",
    "adaptive_meta",
]

REGIMES = [
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

# Color palette for consistent algorithm representation
ALGORITHM_COLORS = {
    "round_robin": "#4e79a7",
    "least_connections": "#59a14f",
    "ip_hash": "#9c755f",
    "logistic_regression": "#f28e2b",
    "random_forest": "#edc948",
    "decision_tree": "#76b7b2",
    "svm": "#b07aa1",
    "xgboost": "#ff9da7",
    "adaptive_policy": "#e15759",
    "adaptive_meta": "#86bc86",
}


def compute_jains_fairness(counts: List[int]) -> float:
    """Compute Jain's Fairness Index for server request distribution:
    J = (sum(x_i))^2 / (n * sum(x_i^2))
    """
    if not counts or sum(counts) == 0:
        return 1.0
    n = len(counts)
    sum_x = sum(counts)
    sum_x_sq = sum(x**2 for x in counts)
    if sum_x_sq == 0:
        return 1.0
    return round((sum_x**2) / (n * sum_x_sq), 4)


class Phase14Analyzer:
    """Analyzes Phase 14 benchmark raw data and generates statistical summaries and plots."""

    def __init__(self, raw_csv_path: str = RAW_CSV_PATH):
        self.raw_csv_path = raw_csv_path
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
        self.df = pd.read_csv(self.raw_csv_path)
        logger.info("Loaded %d raw observations from %s", len(self.df), self.raw_csv_path)

        # Separate core benchmark from priority extension
        self.core_df = self.df[self.df["is_priority_subset"] == False].copy()
        self.priority_df = self.df[self.df["is_priority_subset"] == True].copy()
        logger.info("Core benchmark: %d rows, Priority subset: %d rows", len(self.core_df), len(self.priority_df))

    def generate_summary(self) -> pd.DataFrame:
        """Compute aggregated summary metrics across algorithms and regimes."""
        summary_rows = []

        # 1. Per Algorithm x Regime summary
        for alg in ALGORITHMS:
            for regime in REGIMES:
                sub = self.core_df[(self.core_df["algorithm"] == alg) & (self.core_df["regime"] == regime)]
                if sub.empty:
                    continue
                row = self._compute_group_metrics(sub, group_name=alg, regime=regime)
                summary_rows.append(row)

        # 2. Overall Per Algorithm summary
        for alg in ALGORITHMS:
            sub = self.core_df[self.core_df["algorithm"] == alg]
            if sub.empty:
                continue
            row = self._compute_group_metrics(sub, group_name=alg, regime="OVERALL")
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
        logger.info("Saved summary table to %s (%d rows)", SUMMARY_CSV_PATH, len(summary_df))
        return summary_df

    def _compute_group_metrics(self, sub: pd.DataFrame, group_name: str, regime: str) -> Dict[str, Any]:
        """Compute metrics dictionary for a given data slice."""
        latencies = sub["response_time_ms"].values
        overheads = sub["routing_overhead_ms"].values
        total_req = len(sub)
        success_req = int(sub["success"].sum())
        error_rate = round(((total_req - success_req) / total_req) * 100.0, 2)

        # Server distributions and Jain's index
        b_counts = sub["backend_server"].value_counts().values.tolist()
        jains_index = compute_jains_fairness(b_counts)

        # Routing quality
        optimal_match = (sub["suboptimal_routing"] == 0).sum()
        routing_acc = round((optimal_match / total_req) * 100.0, 2)
        suboptimal_rate = round(100.0 - routing_acc, 2)

        # Macro F1 vs optimal server
        y_true = sub["optimal_server"].astype(str)
        # For prediction, use ml_predicted_server if available, else backend_server mapped to label
        port_to_label = {
            "http://127.0.0.1:8001": "server-1",
            "http://127.0.0.1:8002": "server-2",
            "http://127.0.0.1:8003": "server-3",
        }
        y_pred = sub["ml_predicted_server"].fillna("").replace("", np.nan)
        if y_pred.isna().all():
            y_pred = sub["backend_server"].map(port_to_label).fillna("server-1")
        else:
            y_pred = y_pred.fillna(sub["backend_server"].map(port_to_label)).fillna("server-1")
        
        try:
            macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)) * 100.0, 2)
        except Exception:
            macro_f1 = 0.0

        # Throughput
        # Approximate duration by timestamp span or count / sum
        t_span = max(0.1, sub["timestamp"].max() - sub["timestamp"].min())
        throughput_attempted = round(total_req / t_span, 2)
        throughput_completed = round(success_req / t_span, 2)

        # Fallback rate
        fallback_count = int(sub["ml_fallback"].sum())
        fallback_rate = round((fallback_count / total_req) * 100.0, 2)

        return {
            "algorithm": group_name,
            "regime": regime,
            "total_requests": total_req,
            "successful_requests": success_req,
            "error_rate_pct": error_rate,
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_std_ms": round(float(np.std(latencies)), 2),
            "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
            "latency_p90_ms": round(float(np.percentile(latencies, 90)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "latency_max_ms": round(float(np.max(latencies)), 2),
            "routing_overhead_mean_ms": round(float(np.mean(overheads)), 4),
            "routing_overhead_p95_ms": round(float(np.percentile(overheads, 95)), 4),
            "throughput_attempted_rps": throughput_attempted,
            "throughput_completed_rps": throughput_completed,
            "routing_accuracy_pct": routing_acc,
            "routing_macro_f1_pct": macro_f1,
            "suboptimal_routing_rate_pct": suboptimal_rate,
            "jains_fairness_index": jains_index,
            "fallback_count": fallback_count,
            "fallback_rate_pct": fallback_rate,
        }

    def compute_statistical_tests(self) -> Dict[str, Any]:
        """Perform matched paired hypothesis tests comparing each ML/adaptive strategy
        against Round Robin and Least Connections baselines.
        """
        baselines = ["round_robin", "least_connections"]
        candidates = [alg for alg in ALGORITHMS if alg not in baselines]

        # Group data by (regime, run_idx) to form matched paired observations
        run_means = self.core_df.groupby(["algorithm", "regime", "run_idx"])["response_time_ms"].mean().reset_index()

        results: Dict[str, Any] = {"pairwise_comparisons": {}, "summary": {}}

        for cand in candidates:
            cand_data = run_means[run_means["algorithm"] == cand].sort_values(["regime", "run_idx"])

            for base in baselines:
                base_data = run_means[run_means["algorithm"] == base].sort_values(["regime", "run_idx"])

                # Merge on regime and run_idx to guarantee exact matching
                merged = pd.merge(
                    cand_data,
                    base_data,
                    on=["regime", "run_idx"],
                    suffixes=("_cand", "_base"),
                )

                if len(merged) < 5:
                    continue

                diffs = merged["response_time_ms_cand"] - merged["response_time_ms_base"]
                n = len(diffs)
                mean_diff = float(np.mean(diffs))
                std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

                # Paired t-test
                t_stat, p_val = stats.ttest_rel(merged["response_time_ms_cand"], merged["response_time_ms_base"])

                # Wilcoxon signed-rank test
                try:
                    w_stat, w_pval = stats.wilcoxon(diffs)
                except Exception:
                    w_stat, w_pval = 0.0, 1.0

                # Cohen's d effect size
                cohens_d = (mean_diff / std_diff) if std_diff > 0 else 0.0

                # 95% Confidence Interval
                se = (std_diff / math.sqrt(n)) if n > 0 else 0.0
                t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
                ci_lower = mean_diff - t_crit * se
                ci_upper = mean_diff + t_crit * se

                pair_key = f"{cand}_vs_{base}"
                results["pairwise_comparisons"][pair_key] = {
                    "candidate": cand,
                    "baseline": base,
                    "n_pairs": n,
                    "mean_latency_candidate": round(float(merged["response_time_ms_cand"].mean()), 3),
                    "mean_latency_baseline": round(float(merged["response_time_ms_base"].mean()), 3),
                    "mean_difference_ms": round(mean_diff, 3),
                    "std_difference_ms": round(std_diff, 3),
                    "ci_95_lower": round(ci_lower, 3),
                    "ci_95_upper": round(ci_upper, 3),
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "wilcoxon_p_value": round(float(w_pval), 6),
                    "cohens_d": round(float(cohens_d), 4),
                    "is_significant_at_05": bool(p_val < 0.05),
                    "advantage": "candidate" if (p_val < 0.05 and mean_diff < 0) else ("baseline" if (p_val < 0.05 and mean_diff > 0) else "neutral"),
                }

        with open(STATS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved statistical analysis to %s", STATS_JSON_PATH)
        return results

    def generate_all_plots(self) -> List[str]:
        """Generate all 15 publication figures for Phase 14."""
        plots = []
        plots.append(self.plot_01_end_to_end_latency())
        plots.append(self.plot_02_throughput())
        plots.append(self.plot_03_tail_latency())
        plots.append(self.plot_04_latency_cdf())
        plots.append(self.plot_05_routing_overhead())
        plots.append(self.plot_06_accuracy_vs_latency())
        plots.append(self.plot_07_server_utilization())
        plots.append(self.plot_08_scenario_performance())
        plots.append(self.plot_09_high_stress_performance())
        plots.append(self.plot_10_burst_timeline())
        plots.append(self.plot_11_adaptive_selection_distribution())
        plots.append(self.plot_12_significance_heatmap())
        plots.append(self.plot_13_cost_benefit_frontier())
        plots.append(self.plot_14_priority_deadline())
        plots.append(self.plot_15_final_ranking())
        return plots

    # 1. 01_end_to_end_latency_comparison.png
    def plot_01_end_to_end_latency(self) -> str:
        plt.figure(figsize=(10, 5))
        overall = self.core_df.groupby("algorithm")["response_time_ms"].agg(["mean", "median"]).loc[ALGORITHMS]
        x = np.arange(len(ALGORITHMS))
        width = 0.35
        plt.bar(x - width/2, overall["mean"], width, label="Mean Latency (ms)", color="#4e79a7")
        plt.bar(x + width/2, overall["median"], width, label="Median P50 (ms)", color="#59a14f")
        plt.xticks(x, [a.replace("_", "\n") for a in ALGORITHMS], fontsize=9)
        plt.ylabel("Response Time (ms)")
        plt.title("Phase 14: End-to-End Latency Comparison Across 10 Routing Strategies")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "01_end_to_end_latency_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 2. 02_throughput_comparison.png
    def plot_02_throughput(self) -> str:
        plt.figure(figsize=(10, 5))
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        over = summary[summary["regime"] == "OVERALL"].set_index("algorithm").loc[ALGORITHMS]
        x = np.arange(len(ALGORITHMS))
        width = 0.35
        plt.bar(x - width/2, over["throughput_attempted_rps"], width, label="Attempted (req/s)", color="#76b7b2")
        plt.bar(x + width/2, over["throughput_completed_rps"], width, label="Completed (req/s)", color="#4e79a7")
        plt.xticks(x, [a.replace("_", "\n") for a in ALGORITHMS], fontsize=9)
        plt.ylabel("Throughput (Requests / Second)")
        plt.title("Phase 14: Throughput Comparison (Attempted vs Completed)")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "02_throughput_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 3. 03_p95_p99_tail_latency.png
    def plot_03_tail_latency(self) -> str:
        plt.figure(figsize=(10, 5))
        p95 = self.core_df.groupby("algorithm")["response_time_ms"].apply(lambda s: np.percentile(s, 95)).loc[ALGORITHMS]
        p99 = self.core_df.groupby("algorithm")["response_time_ms"].apply(lambda s: np.percentile(s, 99)).loc[ALGORITHMS]
        x = np.arange(len(ALGORITHMS))
        width = 0.35
        plt.bar(x - width/2, p95, width, label="P95 Tail Latency (ms)", color="#f28e2b")
        plt.bar(x + width/2, p99, width, label="P99 Tail Latency (ms)", color="#e15759")
        plt.xticks(x, [a.replace("_", "\n") for a in ALGORITHMS], fontsize=9)
        plt.ylabel("Tail Latency (ms)")
        plt.title("Phase 14: Tail Latency Profile (P95 vs P99) Across Algorithms")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "03_p95_p99_tail_latency.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 4. 04_latency_cdf_by_algorithm.png
    def plot_04_latency_cdf(self) -> str:
        plt.figure(figsize=(10, 6))
        for alg in ALGORITHMS:
            vals = np.sort(self.core_df[self.core_df["algorithm"] == alg]["response_time_ms"].values)
            y = np.linspace(0, 1, len(vals))
            plt.plot(vals, y, label=alg.replace("_", " "), color=ALGORITHM_COLORS.get(alg, "#333"), linewidth=1.8)
        plt.xlabel("Response Time (ms)")
        plt.ylabel("Cumulative Probability (CDF)")
        plt.title("Phase 14: Empirical Latency CDF by Routing Strategy")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "04_latency_cdf_by_algorithm.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 5. 05_routing_overhead_comparison.png
    def plot_05_routing_overhead(self) -> str:
        plt.figure(figsize=(10, 5))
        overheads = [self.core_df[self.core_df["algorithm"] == alg]["routing_overhead_ms"].values for alg in ALGORITHMS]
        plt.boxplot(overheads, labels=[a.replace("_", "\n") for a in ALGORITHMS], showfliers=False, patch_artist=True)
        plt.ylabel("Routing Overhead (ms)")
        plt.title("Phase 14: Measured Routing Decision Overhead Across Algorithms")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "05_routing_overhead_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 6. 06_accuracy_vs_latency_tradeoff.png
    def plot_06_accuracy_vs_latency(self) -> str:
        plt.figure(figsize=(9, 6))
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        over = summary[summary["regime"] == "OVERALL"].set_index("algorithm").loc[ALGORITHMS]
        for alg in ALGORITHMS:
            row = over.loc[alg]
            plt.scatter(
                row["routing_accuracy_pct"],
                row["latency_mean_ms"],
                s=120,
                color=ALGORITHM_COLORS.get(alg, "#333"),
                label=alg.replace("_", " "),
                edgecolor="black",
            )
            plt.annotate(
                alg.replace("_", " "),
                (row["routing_accuracy_pct"] + 0.5, row["latency_mean_ms"] + 0.1),
                fontsize=8,
            )
        plt.xlabel("Routing Classification Accuracy (% vs Empirical Optimal)")
        plt.ylabel("Mean End-to-End Latency (ms)")
        plt.title("Phase 14: Classification Accuracy vs System Latency Tradeoff")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "06_accuracy_vs_latency_tradeoff.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 7. 07_server_utilization_imbalance.png
    def plot_07_server_utilization(self) -> str:
        plt.figure(figsize=(10, 5))
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        over = summary[summary["regime"] == "OVERALL"].set_index("algorithm").loc[ALGORITHMS]
        x = np.arange(len(ALGORITHMS))
        plt.bar(x, over["jains_fairness_index"], color="#59a14f", width=0.5, edgecolor="black")
        plt.xticks(x, [a.replace("_", "\n") for a in ALGORITHMS], fontsize=9)
        plt.ylim(0, 1.05)
        plt.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Perfect Fairness (1.0)")
        plt.ylabel("Jain's Fairness Index (1.0 = Ideal Balance)")
        plt.title("Phase 14: Server Load Balance & Request Distribution Fairness")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "07_server_utilization_imbalance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 8. 08_scenario_performance_radar.png
    def plot_08_scenario_performance(self) -> str:
        # Heatmap of Mean Latency across Regimes x Algorithms
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        pivot = summary[summary["regime"] != "OVERALL"].pivot(index="regime", columns="algorithm", values="latency_mean_ms")
        pivot = pivot.reindex(index=REGIMES, columns=ALGORITHMS)
        plt.figure(figsize=(11, 7))
        plt.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        plt.colorbar(label="Mean Latency (ms)")
        plt.xticks(range(len(ALGORITHMS)), [a.replace("_", "\n") for a in ALGORITHMS], fontsize=8)
        plt.yticks(range(len(REGIMES)), [r.replace("_", " ").title() for r in REGIMES], fontsize=9)
        for i in range(len(REGIMES)):
            for j in range(len(ALGORITHMS)):
                val = pivot.values[i, j]
                plt.text(j, i, f"{val:.1f}", ha="center", va="center", color="black" if val < pivot.values.max()*0.7 else "white", fontsize=7.5)
        plt.title("Phase 14: Performance Heatmap across 9 Operational Regimes (Mean Latency ms)")
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "08_scenario_performance_radar.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 9. 09_high_stress_contention_performance.png
    def plot_09_high_stress_performance(self) -> str:
        stress_regimes = ["queue_contention", "backend_imbalance", "high_load"]
        sub = self.core_df[self.core_df["regime"].isin(stress_regimes)]
        plt.figure(figsize=(10, 5))
        agg = sub.groupby("algorithm")["response_time_ms"].agg(["mean", "std"]).loc[ALGORITHMS]
        x = np.arange(len(ALGORITHMS))
        plt.bar(x, agg["mean"], yerr=agg["std"], capsize=4, color="#e15759", width=0.5, edgecolor="black")
        plt.xticks(x, [a.replace("_", "\n") for a in ALGORITHMS], fontsize=9)
        plt.ylabel("Mean Latency under Stress (ms)")
        plt.title("Phase 14: High Stress & Contention Performance (Contention, Imbalance, High Load)")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "09_high_stress_contention_performance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 10. 10_burst_recovery_timeline.png
    def plot_10_burst_timeline(self) -> str:
        plt.figure(figsize=(11, 5))
        burst_df = self.core_df[self.core_df["regime"] == "burst_load"]
        selected_algs = ["round_robin", "least_connections", "svm", "adaptive_policy"]
        for alg in selected_algs:
            sub = burst_df[(burst_df["algorithm"] == alg) & (burst_df["run_idx"] == 1)].sort_values("request_id")
            if not sub.empty:
                plt.plot(range(len(sub)), sub["response_time_ms"].values, label=alg.replace("_", " "), linewidth=1.5, marker="o", markersize=3)
        plt.xlabel("Request Sequence Index in Burst")
        plt.ylabel("Response Time (ms)")
        plt.title("Phase 14: Burst Load Response Timeline & System Recovery Dynamics")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "10_burst_recovery_timeline.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 11. 11_adaptive_model_selection_distribution.png
    def plot_11_adaptive_selection_distribution(self) -> str:
        plt.figure(figsize=(9, 5))
        adaptive_sub = self.core_df[self.core_df["algorithm"].isin(["adaptive_policy", "adaptive_meta"])]
        counts = adaptive_sub.groupby(["algorithm", "selected_model"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="tab10", edgecolor="black")
        plt.xticks([0, 1], ["Adaptive Policy", "Adaptive Meta"], rotation=0)
        plt.ylabel("Selected Frequency Count")
        plt.title("Phase 14: Model Selection Frequency by Adaptive Strategies")
        plt.legend(title="Candidate Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "11_adaptive_model_selection_distribution.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 12. 12_statistical_significance_heatmap.png
    def plot_12_significance_heatmap(self) -> str:
        with open(STATS_JSON_PATH, "r", encoding="utf-8") as f:
            stats_data = json.load(f)["pairwise_comparisons"]
        cands = [a for a in ALGORITHMS if a not in ("round_robin", "least_connections")]
        bases = ["round_robin", "least_connections"]
        matrix = np.zeros((len(cands), len(bases)))
        for i, c in enumerate(cands):
            for j, b in enumerate(bases):
                key = f"{c}_vs_{b}"
                matrix[i, j] = stats_data.get(key, {}).get("cohens_d", 0.0)

        plt.figure(figsize=(7, 6))
        plt.imshow(matrix, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="Cohen's d (Negative = Candidate Faster than Baseline)")
        plt.xticks(range(len(bases)), [b.replace("_", " ").title() for b in bases])
        plt.yticks(range(len(cands)), [c.replace("_", " ").title() for c in cands])
        for i in range(len(cands)):
            for j in range(len(bases)):
                key = f"{cands[i]}_vs_{bases[j]}"
                pval = stats_data.get(key, {}).get("p_value", 1.0)
                plt.text(j, i, f"d={matrix[i, j]:.2f}\np={pval:.3f}", ha="center", va="center", fontsize=8)
        plt.title("Phase 14: Statistical Effect Size (Cohen's d) vs Baselines")
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "12_statistical_significance_heatmap.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 13. 13_cost_benefit_frontier.png
    def plot_13_cost_benefit_frontier(self) -> str:
        plt.figure(figsize=(9, 6))
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        over = summary[summary["regime"] == "OVERALL"].set_index("algorithm").loc[ALGORITHMS]
        for alg in ALGORITHMS:
            row = over.loc[alg]
            plt.scatter(
                row["routing_overhead_mean_ms"],
                row["latency_mean_ms"],
                s=140,
                color=ALGORITHM_COLORS.get(alg, "#333"),
                edgecolor="black",
            )
            plt.annotate(
                alg.replace("_", " "),
                (row["routing_overhead_mean_ms"] + 0.005, row["latency_mean_ms"] + 0.05),
                fontsize=8,
            )
        plt.xlabel("Routing Overhead (ms)")
        plt.ylabel("Mean Response Time (ms)")
        plt.title("Phase 14: Cost-Benefit Frontier (Routing Overhead vs End-to-End Latency)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "13_cost_benefit_frontier.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 14. 14_priority_deadline_performance.png
    def plot_14_priority_deadline(self) -> str:
        plt.figure(figsize=(10, 5))
        if not self.priority_df.empty:
            prio_summary = self.priority_df.groupby(["algorithm", "priority"])["deadline_met"].mean() * 100.0
            unstacked = prio_summary.unstack(fill_value=100.0)
            unstacked.plot(kind="bar", ax=plt.gca(), colormap="viridis", edgecolor="black")
            plt.ylabel("Deadline Adherence Rate (%)")
            plt.title("Phase 14: Deadline Adherence Rate by Priority Level")
            plt.legend(title="Priority")
            plt.xticks(rotation=15, ha="right", fontsize=9)
            plt.grid(axis="y", linestyle="--", alpha=0.5)
        else:
            plt.text(0.5, 0.5, "Priority evaluation subset empty", ha="center", va="center")
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "14_priority_deadline_performance.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # 15. 15_final_algorithm_ranking.png
    def plot_15_final_ranking(self) -> str:
        # Multi-metric composite score: lower latency rank + lower overhead rank + lower suboptimal rate rank
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        over = summary[summary["regime"] == "OVERALL"].set_index("algorithm").loc[ALGORITHMS]
        r_lat = over["latency_mean_ms"].rank()
        r_tail = over["latency_p95_ms"].rank()
        r_ovh = over["routing_overhead_mean_ms"].rank()
        r_sub = over["suboptimal_routing_rate_pct"].rank()
        r_fair = (-over["jains_fairness_index"]).rank()

        composite = (r_lat * 0.35 + r_tail * 0.25 + r_ovh * 0.15 + r_sub * 0.15 + r_fair * 0.10).sort_values()

        plt.figure(figsize=(10, 5))
        colors = ["#59a14f" if i == 0 else "#4e79a7" for i in range(len(composite))]
        plt.barh(range(len(composite)), composite.values, color=colors, edgecolor="black")
        plt.yticks(range(len(composite)), [a.replace("_", " ").title() for a in composite.index])
        plt.xlabel("Composite Rank Score (Lower = Better)")
        plt.title("Phase 14: Final Multi-Metric Algorithm Ranking")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(OUTPUT_PLOTS_DIR, "15_final_algorithm_ranking.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def print_markdown_tables(self) -> None:
        """Print TABLES A through E formatted in GitHub markdown for reporting."""
        summary = pd.read_csv(SUMMARY_CSV_PATH)
        over = summary[summary["regime"] == "OVERALL"].set_index("algorithm").loc[ALGORITHMS]

        print("\n### TABLE A: End-to-End Latency Summary Across 10 Routing Strategies")
        print("| Routing Algorithm | Mean (ms) | Std (ms) | P50 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for alg in ALGORITHMS:
            r = over.loc[alg]
            print(f"| `{alg}` | {r['latency_mean_ms']:.2f} | {r['latency_std_ms']:.2f} | {r['latency_p50_ms']:.2f} | {r['latency_p90_ms']:.2f} | {r['latency_p95_ms']:.2f} | {r['latency_p99_ms']:.2f} | {r['latency_max_ms']:.2f} |")

        print("\n### TABLE B: System Throughput and Reliability Metrics")
        print("| Routing Algorithm | Attempted (req/s) | Completed (req/s) | Error Rate (%) | Fallback Count | Fallback Rate (%) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for alg in ALGORITHMS:
            r = over.loc[alg]
            print(f"| `{alg}` | {r['throughput_attempted_rps']:.2f} | {r['throughput_completed_rps']:.2f} | {r['error_rate_pct']:.2f}% | {int(r['fallback_count'])} | {r['fallback_rate_pct']:.2f}% |")

        print("\n### TABLE C: Routing Decision Overhead & Quality")
        print("| Routing Algorithm | Mean Overhead (ms) | P95 Overhead (ms) | Classification Acc (%) | Macro F1 (%) | Suboptimal Rate (%) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for alg in ALGORITHMS:
            r = over.loc[alg]
            print(f"| `{alg}` | {r['routing_overhead_mean_ms']:.4f} | {r['routing_overhead_p95_ms']:.4f} | {r['routing_accuracy_pct']:.2f}% | {r['routing_macro_f1_pct']:.2f}% | {r['suboptimal_routing_rate_pct']:.2f}% |")

        print("\n### TABLE D: Statistical Hypothesis Testing (vs Baselines)")
        with open(STATS_JSON_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)["pairwise_comparisons"]
        print("| Comparison | Mean Diff (ms) | 95% CI (ms) | t-stat | p-value | Cohen's d | Significant (p<0.05) |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k, v in st.items():
            print(f"| `{k}` | {v['mean_difference_ms']:+.3f} | [{v['ci_95_lower']:.3f}, {v['ci_95_upper']:.3f}] | {v['t_statistic']:.3f} | {v['p_value']:.4f} | {v['cohens_d']:.3f} | {v['is_significant_at_05']} |")


if __name__ == "__main__":
    analyzer = Phase14Analyzer()
    analyzer.generate_summary()
    analyzer.compute_statistical_tests()
    analyzer.generate_all_plots()
    analyzer.print_markdown_tables()

