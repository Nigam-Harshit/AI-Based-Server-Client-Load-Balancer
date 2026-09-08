"""Phase 10 Priority & Deadline Aware Statistical Analysis and Visualization Suite.

Computes statistical comparisons between standard ML routing and Priority & Deadline-Aware
ML routing across scenarios A-G, generating quantitative metrics and publication-quality plots.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("benchmarks.phase10_analysis")

DATA_DIR = "data/phase10"
PLOT_DIR = "docs/plots/phase10"

SCENARIO_LABELS = {
    "scenario_a_equal_priority": "A: Equal Priority",
    "scenario_b_mixed_priority": "B: Mixed Priority",
    "scenario_c_tight_deadlines": "C: Tight Deadlines",
    "scenario_d_priority_deadline_conflict": "D: Priority-Deadline Conflict",
    "scenario_e_backend_stress": "E: Backend Stress",
    "scenario_f_backend_failure": "F: Backend Failure",
    "scenario_g_queue_contention": "G: Queue Contention",
}

ALGO_COLORS = {
    "ml": "#4285F4",           # Google Blue
    "priority_ml": "#0F9D58",  # Google Green
}


class Phase10Analyzer:
    """Performs statistical analysis and creates charts for Phase 10 benchmarks."""

    def __init__(self, data_dir: str = DATA_DIR, plot_dir: str = PLOT_DIR):
        self.data_dir = data_dir
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

        self.raw_path = os.path.join(self.data_dir, "raw_results.csv")
        self.summaries_path = os.path.join(self.data_dir, "run_summaries.csv")
        self.stats_output_path = os.path.join(self.data_dir, "statistical_analysis.json")

        self.df_raw = pd.read_csv(self.raw_path) if os.path.exists(self.raw_path) else pd.DataFrame()
        self.df_summaries = pd.read_csv(self.summaries_path) if os.path.exists(self.summaries_path) else pd.DataFrame()

    def generate_all(self) -> Dict[str, Any]:
        """Compute all statistics and generate plots."""
        if self.df_summaries.empty:
            logger.warning("No summary data found in %s", self.summaries_path)
            return {}

        logger.info("Computing Phase 10 statistical summaries...")
        stats_summary = self.compute_statistical_summaries()

        logger.info("Computing Phase 10 statistical significance...")
        significance = self.compute_statistical_significance()

        logger.info("Generating Phase 10 research plots...")
        self.plot_throughput_comparison()
        self.plot_p50_latency_comparison()
        self.plot_deadline_violation_rate()
        self.plot_latency_by_priority_tier()
        self.plot_override_frequency()

        combined = {
            "statistics": stats_summary,
            "significance": significance,
        }

        with open(self.stats_output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)

        logger.info("Analysis complete. Saved report to %s and plots to %s", self.stats_output_path, self.plot_dir)
        return combined

    def compute_statistical_summaries(self) -> Dict[str, Any]:
        """Compute descriptive statistics grouped by scenario and algorithm."""
        metrics = [
            "throughput_rps",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "avg_latency_ms",
            "deadline_violation_rate",
            "avg_lateness_ms",
            "priority_overrides",
            "high_critical_avg_latency_ms",
            "low_normal_avg_latency_ms",
            "success_rate",
        ]

        grouped = self.df_summaries.groupby(["scenario", "algorithm"])
        summary_dict = {}

        for (scen, algo), group in grouped:
            key = f"{scen}::{algo}"
            summary_dict[key] = {
                "scenario": scen,
                "algorithm": algo,
                "label": SCENARIO_LABELS.get(scen, scen),
                "repetitions": len(group),
            }
            for m in metrics:
                vals = group[m].dropna().values
                if len(vals) > 0:
                    summary_dict[key][m] = {
                        "mean": round(float(np.mean(vals)), 2),
                        "median": round(float(np.median(vals)), 2),
                        "std": round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else 0.0,
                        "min": round(float(np.min(vals)), 2),
                        "max": round(float(np.max(vals)), 2),
                    }
                else:
                    summary_dict[key][m] = None

        return summary_dict

    def compute_statistical_significance(self) -> Dict[str, Any]:
        """Calculate Welch's t-test between ML and Priority ML across scenarios."""
        comparisons = {}
        scenarios = self.df_summaries["scenario"].unique()

        for scen in scenarios:
            sub = self.df_summaries[self.df_summaries["scenario"] == scen]
            ml_sub = sub[sub["algorithm"] == "ml"]
            pml_sub = sub[sub["algorithm"] == "priority_ml"]

            if ml_sub.empty or pml_sub.empty:
                continue

            # Latency (P50)
            ml_p50 = ml_sub["p50_latency_ms"].values
            pml_p50 = pml_sub["p50_latency_ms"].values
            t_p50, p_p50 = stats.ttest_ind(pml_p50, ml_p50, equal_var=False) if len(ml_p50) > 1 else (0.0, 1.0)

            # Deadline violation rate
            ml_viol = ml_sub["deadline_violation_rate"].values
            pml_viol = pml_sub["deadline_violation_rate"].values
            t_viol, p_viol = stats.ttest_ind(pml_viol, ml_viol, equal_var=False) if len(ml_viol) > 1 else (0.0, 1.0)

            # High/Critical latency
            ml_hc = ml_sub["high_critical_avg_latency_ms"].dropna().values
            pml_hc = pml_sub["high_critical_avg_latency_ms"].dropna().values
            t_hc, p_hc = stats.ttest_ind(pml_hc, ml_hc, equal_var=False) if (len(ml_hc) > 1 and len(pml_hc) > 1) else (0.0, 1.0)

            comparisons[scen] = {
                "label": SCENARIO_LABELS.get(scen, scen),
                "p50_latency_diff_mean": round(float(np.mean(pml_p50) - np.mean(ml_p50)), 2),
                "p50_welch_t": round(float(t_p50), 3) if not np.isnan(t_p50) else None,
                "p50_p_value": round(float(p_p50), 4) if not np.isnan(p_p50) else None,
                "violation_rate_diff_mean": round(float(np.mean(pml_viol) - np.mean(ml_viol)), 2),
                "violation_welch_t": round(float(t_viol), 3) if not np.isnan(t_viol) else None,
                "violation_p_value": round(float(p_viol), 4) if not np.isnan(p_viol) else None,
                "high_critical_latency_diff_mean": round(float(np.mean(pml_hc) - np.mean(ml_hc)), 2) if (len(ml_hc) > 0 and len(pml_hc) > 0) else None,
            }

        return comparisons

    def plot_throughput_comparison(self) -> None:
        """Plot throughput comparison across scenarios."""
        plt.figure(figsize=(12, 6))
        scenarios = list(SCENARIO_LABELS.keys())
        x = np.arange(len(scenarios))
        width = 0.35

        ml_means = []
        ml_errs = []
        pml_means = []
        pml_errs = []

        for scen in scenarios:
            sub = self.df_summaries[self.df_summaries["scenario"] == scen]
            ml_vals = sub[sub["algorithm"] == "ml"]["throughput_rps"].values
            pml_vals = sub[sub["algorithm"] == "priority_ml"]["throughput_rps"].values

            ml_means.append(np.mean(ml_vals) if len(ml_vals) > 0 else 0)
            ml_errs.append(np.std(ml_vals) if len(ml_vals) > 0 else 0)
            pml_means.append(np.mean(pml_vals) if len(pml_vals) > 0 else 0)
            pml_errs.append(np.std(pml_vals) if len(pml_vals) > 0 else 0)

        plt.bar(x - width/2, ml_means, width, yerr=ml_errs, capsize=4, label="Standard ML", color=ALGO_COLORS["ml"], edgecolor="black", alpha=0.9)
        plt.bar(x + width/2, pml_means, width, yerr=pml_errs, capsize=4, label="Priority ML", color=ALGO_COLORS["priority_ml"], edgecolor="black", alpha=0.9)

        plt.title("Phase 10: Throughput Comparison by Workload Scenario (RPS)", fontsize=14, fontweight="bold")
        plt.xlabel("Scenario", fontsize=12)
        plt.ylabel("Throughput (Requests / sec)", fontsize=12)
        plt.xticks(x, [SCENARIO_LABELS.get(s, s).split(": ")[1] for s in scenarios], rotation=25, ha="right")
        plt.legend(frameon=True, facecolor="white", edgecolor="gray")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        out = os.path.join(self.plot_dir, "01_p10_throughput_comparison.png")
        plt.savefig(out, dpi=300)
        plt.close()

    def plot_p50_latency_comparison(self) -> None:
        """Plot median (P50) latency comparison across scenarios."""
        plt.figure(figsize=(12, 6))
        scenarios = list(SCENARIO_LABELS.keys())
        x = np.arange(len(scenarios))
        width = 0.35

        ml_means, ml_errs = [], []
        pml_means, pml_errs = [], []

        for scen in scenarios:
            sub = self.df_summaries[self.df_summaries["scenario"] == scen]
            ml_vals = sub[sub["algorithm"] == "ml"]["p50_latency_ms"].values
            pml_vals = sub[sub["algorithm"] == "priority_ml"]["p50_latency_ms"].values

            ml_means.append(np.mean(ml_vals) if len(ml_vals) > 0 else 0)
            ml_errs.append(np.std(ml_vals) if len(ml_vals) > 0 else 0)
            pml_means.append(np.mean(pml_vals) if len(pml_vals) > 0 else 0)
            pml_errs.append(np.std(pml_vals) if len(pml_vals) > 0 else 0)

        plt.bar(x - width/2, ml_means, width, yerr=ml_errs, capsize=4, label="Standard ML", color=ALGO_COLORS["ml"], edgecolor="black", alpha=0.9)
        plt.bar(x + width/2, pml_means, width, yerr=pml_errs, capsize=4, label="Priority ML", color=ALGO_COLORS["priority_ml"], edgecolor="black", alpha=0.9)

        plt.title("Phase 10: Median Latency (P50) Comparison (ms)", fontsize=14, fontweight="bold")
        plt.xlabel("Scenario", fontsize=12)
        plt.ylabel("P50 Latency (ms)", fontsize=12)
        plt.xticks(x, [SCENARIO_LABELS.get(s, s).split(": ")[1] for s in scenarios], rotation=25, ha="right")
        plt.legend(frameon=True, facecolor="white", edgecolor="gray")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        out = os.path.join(self.plot_dir, "02_p10_p50_latency_comparison.png")
        plt.savefig(out, dpi=300)
        plt.close()

    def plot_deadline_violation_rate(self) -> None:
        """Plot deadline violation rate (%) comparison across scenarios."""
        plt.figure(figsize=(12, 6))
        scenarios = list(SCENARIO_LABELS.keys())
        x = np.arange(len(scenarios))
        width = 0.35

        ml_means, ml_errs = [], []
        pml_means, pml_errs = [], []

        for scen in scenarios:
            sub = self.df_summaries[self.df_summaries["scenario"] == scen]
            ml_vals = sub[sub["algorithm"] == "ml"]["deadline_violation_rate"].values
            pml_vals = sub[sub["algorithm"] == "priority_ml"]["deadline_violation_rate"].values

            ml_means.append(np.mean(ml_vals) if len(ml_vals) > 0 else 0)
            ml_errs.append(np.std(ml_vals) if len(ml_vals) > 0 else 0)
            pml_means.append(np.mean(pml_vals) if len(pml_vals) > 0 else 0)
            pml_errs.append(np.std(pml_vals) if len(pml_vals) > 0 else 0)

        plt.bar(x - width/2, ml_means, width, yerr=ml_errs, capsize=4, label="Standard ML", color="#EA4335", edgecolor="black", alpha=0.85)
        plt.bar(x + width/2, pml_means, width, yerr=pml_errs, capsize=4, label="Priority ML", color="#0F9D58", edgecolor="black", alpha=0.85)

        plt.title("Phase 10: Deadline Violation Rate Comparison (%)", fontsize=14, fontweight="bold")
        plt.xlabel("Scenario", fontsize=12)
        plt.ylabel("Violation Rate (%)", fontsize=12)
        plt.xticks(x, [SCENARIO_LABELS.get(s, s).split(": ")[1] for s in scenarios], rotation=25, ha="right")
        plt.legend(frameon=True, facecolor="white", edgecolor="gray")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        out = os.path.join(self.plot_dir, "03_p10_deadline_violation_rate.png")
        plt.savefig(out, dpi=300)
        plt.close()

    def plot_latency_by_priority_tier(self) -> None:
        """Plot response time comparison broken down by request priority tier."""
        if self.df_raw.empty:
            return

        plt.figure(figsize=(10, 6))
        tiers = ["LOW", "NORMAL", "HIGH", "CRITICAL"]
        x = np.arange(len(tiers))
        width = 0.35

        ml_means, ml_errs = [], []
        pml_means, pml_errs = [], []

        for t in tiers:
            ml_data = self.df_raw[(self.df_raw["algorithm"] == "ml") & (self.df_raw["priority"] == t) & (self.df_raw["success"] == True)]["response_time_ms"]
            pml_data = self.df_raw[(self.df_raw["algorithm"] == "priority_ml") & (self.df_raw["priority"] == t) & (self.df_raw["success"] == True)]["response_time_ms"]

            ml_means.append(ml_data.mean() if len(ml_data) > 0 else 0)
            ml_errs.append(ml_data.std() if len(ml_data) > 0 else 0)
            pml_means.append(pml_data.mean() if len(pml_data) > 0 else 0)
            pml_errs.append(pml_data.std() if len(pml_data) > 0 else 0)

        plt.bar(x - width/2, ml_means, width, yerr=ml_errs, capsize=4, label="Standard ML", color=ALGO_COLORS["ml"], edgecolor="black", alpha=0.9)
        plt.bar(x + width/2, pml_means, width, yerr=pml_errs, capsize=4, label="Priority ML", color=ALGO_COLORS["priority_ml"], edgecolor="black", alpha=0.9)

        plt.title("Phase 10: Mean Latency by Priority Tier (ms)", fontsize=14, fontweight="bold")
        plt.xlabel("Request Priority Tier", fontsize=12)
        plt.ylabel("Mean Latency (ms)", fontsize=12)
        plt.xticks(x, tiers)
        plt.legend(frameon=True, facecolor="white", edgecolor="gray")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        out = os.path.join(self.plot_dir, "04_p10_latency_by_priority_tier.png")
        plt.savefig(out, dpi=300)
        plt.close()

    def plot_override_frequency(self) -> None:
        """Plot frequency of ML routing overrides by PriorityDeadlineRouter across scenarios."""
        pml_runs = self.df_summaries[self.df_summaries["algorithm"] == "priority_ml"]
        if pml_runs.empty:
            return

        plt.figure(figsize=(12, 6))
        scenarios = list(SCENARIO_LABELS.keys())
        x = np.arange(len(scenarios))

        means = []
        errs = []
        for scen in scenarios:
            sub = pml_runs[pml_runs["scenario"] == scen]
            vals = sub["priority_overrides"].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            errs.append(np.std(vals) if len(vals) > 0 else 0)

        plt.bar(x, means, yerr=errs, capsize=4, color="#FBBC04", edgecolor="black", alpha=0.9, width=0.5)
        plt.title("Phase 10: Router Overrides Count per Benchmark Run", fontsize=14, fontweight="bold")
        plt.xlabel("Scenario", fontsize=12)
        plt.ylabel("Number of Overridden Decisions", fontsize=12)
        plt.xticks(x, [SCENARIO_LABELS.get(s, s).split(": ")[1] for s in scenarios], rotation=25, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()

        out = os.path.join(self.plot_dir, "05_p10_override_frequency_by_scenario.png")
        plt.savefig(out, dpi=300)
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analyzer = Phase10Analyzer()
    analyzer.generate_all()

