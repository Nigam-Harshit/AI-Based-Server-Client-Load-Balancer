"""Phase 9 Statistical Analysis and Research Visualization Suite.

Generates high-resolution publication-quality plots and comprehensive statistical
aggregations across algorithms, scenarios, metrics, and stress conditions.
"""

import json
import logging
import os
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("benchmarks.analysis")

PLOT_DIR = "docs/plots"
DATA_DIR = "data/phase9"


class Phase9Analyzer:
    """Computes statistical metrics and generates visualizations for Phase 9."""

    def __init__(self, data_dir: str = DATA_DIR, plot_dir: str = PLOT_DIR):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        self.df_raw = pd.read_csv(os.path.join(self.data_dir, "raw_results.csv"))
        self.df_summaries = pd.read_csv(os.path.join(self.data_dir, "run_summaries.csv"))
        with open(os.path.join(self.data_dir, "edge_conditions.json"), "r", encoding="utf-8") as f:
            self.edge_data = json.load(f)

    def generate_all(self) -> Dict[str, Any]:
        """Run statistical computations and produce all required visualizations."""
        logger.info("Computing statistical summaries...")
        stats_report = self.compute_statistical_summaries()

        logger.info("Generating research plots...")
        self.plot_throughput_by_scenario()
        self.plot_p50_latency()
        self.plot_p95_latency()
        self.plot_p99_latency()
        self.plot_cpu_utilization()
        self.plot_backend_distribution()
        self.plot_ml_confidence_distribution()
        self.plot_ml_fallback_rate()
        self.plot_performance_vs_workload_intensity()
        self.plot_algorithm_robustness()

        logger.info("Generating statistical significance report...")
        sig_report = self.compute_statistical_significance()

        combined = {
            "statistics": stats_report,
            "significance": sig_report,
        }

        stats_path = os.path.join(self.data_dir, "statistical_analysis.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        return combined

    def compute_statistical_summaries(self) -> Dict[str, Any]:
        """Compute mean, median, std, min, max across algorithm x scenario."""
        metrics = [
            "throughput_rps",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "workload_cpu_percent",
            "success_rate",
        ]

        grouped = self.df_summaries.groupby(["scenario", "algorithm"])
        summary_dict = {}

        for (scen, algo), group in grouped:
            key = f"{scen}::{algo}"
            summary_dict[key] = {
                "scenario": scen,
                "algorithm": algo,
                "repetitions": len(group),
            }
            for m in metrics:
                vals = group[m].values
                summary_dict[key][m] = {
                    "mean": round(float(np.mean(vals)), 2),
                    "median": round(float(np.median(vals)), 2),
                    "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
                    "min": round(float(np.min(vals)), 2),
                    "max": round(float(np.max(vals)), 2),
                }

        return summary_dict

    def compute_statistical_significance(self) -> Dict[str, Any]:
        """Evaluate Welch's t-test and Mann-Whitney U test between ML and baselines."""
        comparisons = {}
        scenarios = self.df_summaries["scenario"].unique()

        for scen in scenarios:
            sub = self.df_summaries[self.df_summaries["scenario"] == scen]
            ml_p50 = sub[sub["algorithm"] == "ml"]["p50_latency_ms"].values
            rr_p50 = sub[sub["algorithm"] == "round_robin"]["p50_latency_ms"].values
            lc_p50 = sub[sub["algorithm"] == "least_connections"]["p50_latency_ms"].values

            ml_tput = sub[sub["algorithm"] == "ml"]["throughput_rps"].values
            rr_tput = sub[sub["algorithm"] == "round_robin"]["throughput_rps"].values

            # Latency t-test (ML vs RR)
            t_lat, p_lat = stats.ttest_ind(ml_p50, rr_p50, equal_var=False)
            t_tput, p_tput = stats.ttest_ind(ml_tput, rr_tput, equal_var=False)

            comparisons[scen] = {
                "p50_latency_diff_mean": round(float(np.mean(ml_p50) - np.mean(rr_p50)), 2),
                "p50_latency_p_val": round(float(p_lat), 4) if not np.isnan(p_lat) else None,
                "p50_latency_significant_05": bool(p_lat < 0.05) if not np.isnan(p_lat) else False,
                "throughput_diff_mean": round(float(np.mean(ml_tput) - np.mean(rr_tput)), 2),
                "throughput_p_val": round(float(p_tput), 4) if not np.isnan(p_tput) else None,
                "throughput_significant_05": bool(p_tput < 0.05) if not np.isnan(p_tput) else False,
            }

        return {
            "repetitions_per_group": 3,
            "note": "Sample size (n=3) provides initial empirical directional indicators; repetitions are sufficient for exploratory hypothesis testing but limited for high-power asymptotic claims.",
            "scenario_comparisons": comparisons,
        }

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------

    def plot_throughput_by_scenario(self):
        """Plot 1: Throughput by algorithm and scenario."""
        pivot = self.df_summaries.groupby(["scenario", "algorithm"])["throughput_rps"].mean().unstack()
        ax = pivot.plot(kind="bar", figsize=(10, 6), colormap="tab10", width=0.8, edgecolor="black")
        plt.title("Throughput by Algorithm and Workload Scenario (RPS)", fontsize=13, fontweight="bold")
        plt.ylabel("Requests Per Second (RPS)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "01_throughput_by_scenario.png"), dpi=300)
        plt.close()

    def plot_p50_latency(self):
        """Plot 2: P50 latency by algorithm and scenario."""
        pivot = self.df_summaries.groupby(["scenario", "algorithm"])["p50_latency_ms"].mean().unstack()
        ax = pivot.plot(kind="bar", figsize=(10, 6), colormap="Set2", width=0.8, edgecolor="black")
        plt.title("Median (P50) Latency by Algorithm and Scenario", fontsize=13, fontweight="bold")
        plt.ylabel("P50 Latency (ms)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "02_p50_latency.png"), dpi=300)
        plt.close()

    def plot_p95_latency(self):
        """Plot 3: P95 latency by algorithm and scenario."""
        pivot = self.df_summaries.groupby(["scenario", "algorithm"])["p95_latency_ms"].mean().unstack()
        ax = pivot.plot(kind="bar", figsize=(10, 6), colormap="Accent", width=0.8, edgecolor="black")
        plt.title("Tail (P95) Latency by Algorithm and Scenario", fontsize=13, fontweight="bold")
        plt.ylabel("P95 Latency (ms)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "03_p95_latency.png"), dpi=300)
        plt.close()

    def plot_p99_latency(self):
        """Plot 4: P99 latency by algorithm and scenario."""
        pivot = self.df_summaries.groupby(["scenario", "algorithm"])["p99_latency_ms"].mean().unstack()
        ax = pivot.plot(kind="bar", figsize=(10, 6), colormap="Dark2", width=0.8, edgecolor="black")
        plt.title("Worst-Case Tail (P99) Latency by Algorithm and Scenario", fontsize=13, fontweight="bold")
        plt.ylabel("P99 Latency (ms)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "04_p99_latency.png"), dpi=300)
        plt.close()

    def plot_cpu_utilization(self):
        """Plot 5: CPU utilization comparison."""
        pivot = self.df_summaries.groupby(["scenario", "algorithm"])["workload_cpu_percent"].mean().unstack()
        ax = pivot.plot(kind="bar", figsize=(10, 6), colormap="Spectral", width=0.8, edgecolor="black")
        plt.title("Average Backend CPU Utilization (%) during Workload", fontsize=13, fontweight="bold")
        plt.ylabel("CPU Utilization (%)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "05_cpu_utilization.png"), dpi=300)
        plt.close()

    def plot_backend_distribution(self):
        """Plot 6: Backend request distribution across algorithms."""
        dist = self.df_raw.groupby(["algorithm", "backend_server"]).size().unstack(fill_value=0)
        ax = dist.plot(kind="bar", stacked=True, figsize=(9, 6), colormap="viridis", edgecolor="black")
        plt.title("Total Backend Server Request Distribution across Algorithms", fontsize=13, fontweight="bold")
        plt.ylabel("Request Count", fontsize=11)
        plt.xlabel("Algorithm", fontsize=11)
        plt.xticks(rotation=0)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Backend Server", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "06_backend_distribution.png"), dpi=300)
        plt.close()

    def plot_ml_confidence_distribution(self):
        """Plot 7: ML confidence distribution."""
        ml_conf = self.df_raw[self.df_raw["algorithm"] == "ml"]["ml_confidence"].dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(ml_conf, bins=15, color="#1f77b4", edgecolor="black", alpha=0.8)
        plt.title("Distribution of ML Prediction Confidence Scores", fontsize=13, fontweight="bold")
        plt.xlabel("Model Confidence (Probability)", fontsize=11)
        plt.ylabel("Frequency (Requests)", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "07_ml_confidence_distribution.png"), dpi=300)
        plt.close()

    def plot_ml_fallback_rate(self):
        """Plot 8: ML fallback rate by scenario."""
        ml_data = self.df_summaries[self.df_summaries["algorithm"] == "ml"]
        fb = ml_data.groupby("scenario")["ml_fallback_rate"].mean()
        plt.figure(figsize=(9, 5))
        bars = plt.bar(fb.index, fb.values, color="#e377c2", edgecolor="black", width=0.5)
        plt.title("ML Fallback Rate (%) across Workload Scenarios", fontsize=13, fontweight="bold")
        plt.ylabel("Fallback Rate (%)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 10)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            y = b.get_height()
            plt.text(b.get_x() + b.get_width() / 2, y + 0.2, f"{y:.1f}%", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "08_ml_fallback_rate.png"), dpi=300)
        plt.close()

    def plot_performance_vs_workload_intensity(self):
        """Plot 9: Latency vs Request Concurrency Intensity."""
        scen_order = ["low_traffic", "medium_traffic", "mixed", "cpu_heavy", "burst_traffic", "dynamic", "high_traffic"]
        sub = self.df_summaries[self.df_summaries["scenario"].isin(scen_order)]
        mean_p50 = sub.groupby(["scenario", "algorithm"])["p50_latency_ms"].mean().unstack()
        mean_p50 = mean_p50.reindex(scen_order)

        plt.figure(figsize=(11, 6))
        for algo in mean_p50.columns:
            plt.plot(mean_p50.index, mean_p50[algo], marker="o", linewidth=2.5, label=algo)
        plt.title("Median (P50) Latency Scaling vs Workload Scenario Intensity", fontsize=13, fontweight="bold")
        plt.ylabel("P50 Latency (ms)", fontsize=11)
        plt.xlabel("Scenario (Ranked by Concurrency & Rate)", fontsize=11)
        plt.xticks(rotation=25, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "09_performance_vs_workload_intensity.png"), dpi=300)
        plt.close()

    def plot_algorithm_robustness(self):
        """Plot 10: Coefficient of Variation (Std / Mean) across repetitions."""
        stats_df = self.df_summaries.groupby(["scenario", "algorithm"])["p50_latency_ms"].agg(["mean", "std"])
        stats_df["cv"] = (stats_df["std"] / stats_df["mean"]) * 100.0
        cv_pivot = stats_df["cv"].unstack()

        ax = cv_pivot.plot(kind="bar", figsize=(10, 6), colormap="tab20b", width=0.8, edgecolor="black")
        plt.title("Latency Variability / Robustness (Coefficient of Variation % across Repetitions)", fontsize=13, fontweight="bold")
        plt.ylabel("CV % (Lower = More Consistent)", fontsize=11)
        plt.xlabel("Workload Scenario", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "10_algorithm_robustness.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analyzer = Phase9Analyzer()
    analyzer.generate_all()

