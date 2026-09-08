"""Unit and Integration Tests for Phase 14: End-to-End System Benchmarking and Final Performance Validation.

Contains 26 comprehensive verification tests:
1. Manifest existence and valid schema.
2. Environment metadata completeness.
3. Raw dataset existence and schema completeness.
4. Non-empty observation count.
5. All 10 routing algorithms present in core dataset.
6. All 9 operational regimes present in core dataset.
7. Matched repetition counts (>= 5) across algorithm x regime.
8. Request ID global uniqueness.
9. Latency positivity and finiteness.
10. HTTP status code validity.
11. Summary dataset existence and schema.
12. Latency percentile monotonicity (P50 <= P95 <= P99 <= Max).
13. Error rate bounds [0, 100].
14. Statistical analysis JSON validity.
15. Paired comparison p-value bounds [0, 1].
16. 95% Confidence Interval structural consistency (lower <= mean <= upper).
17. Suboptimal routing rate bounds [0, 100].
18. Jain's Fairness Index bounds [0, 1].
19. Routing overhead neutral measurement (non-negative, present, finite for all algorithms; no assumed ordering).
20. Fallback rate tracking and non-negativity.
21. Priority evaluation isolation and metadata validity.
22. Presence of all 15 generated figures.
23. Non-empty and valid PNG headers for all 15 plots.
24. Historical dataset immutability (Phases 5, 9, 10, 12, 13).
25. Candidate model artifacts integrity in models/.
26. Optimal server label validity.
"""

import json
import os
import unittest
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "phase14")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results", "phase14")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RAW_CSV = os.path.join(DATA_DIR, "phase14_raw.csv")
SUMMARY_CSV = os.path.join(DATA_DIR, "phase14_summary.csv")
MANIFEST_FILE = os.path.join(DATA_DIR, "experiment_manifest.json")
ENV_FILE = os.path.join(DATA_DIR, "environment.json")
STATS_FILE = os.path.join(DATA_DIR, "statistical_analysis.json")

EXPECTED_ALGORITHMS = [
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

EXPECTED_REGIMES = [
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

EXPECTED_PLOTS = [
    "01_end_to_end_latency_comparison.png",
    "02_throughput_comparison.png",
    "03_p95_p99_tail_latency.png",
    "04_latency_cdf_by_algorithm.png",
    "05_routing_overhead_comparison.png",
    "06_accuracy_vs_latency_tradeoff.png",
    "07_server_utilization_imbalance.png",
    "08_scenario_performance_radar.png",
    "09_high_stress_contention_performance.png",
    "10_burst_recovery_timeline.png",
    "11_adaptive_model_selection_distribution.png",
    "12_statistical_significance_heatmap.png",
    "13_cost_benefit_frontier.png",
    "14_priority_deadline_performance.png",
    "15_final_algorithm_ranking.png",
]


class TestPhase14Benchmark(unittest.TestCase):
    """Test suite validating Phase 14 experimental benchmark outputs and statistical rigor."""

    @classmethod
    def setUpClass(cls):
        # Verify files exist or raise informative assertion
        assert os.path.exists(RAW_CSV), f"Missing raw CSV: {RAW_CSV}"
        cls.df_raw = pd.read_csv(RAW_CSV)
        cls.core_df = cls.df_raw[cls.df_raw["is_priority_subset"] == False]
        cls.prio_df = cls.df_raw[cls.df_raw["is_priority_subset"] == True]

        assert os.path.exists(SUMMARY_CSV), f"Missing summary CSV: {SUMMARY_CSV}"
        cls.df_summary = pd.read_csv(SUMMARY_CSV)

        assert os.path.exists(MANIFEST_FILE), f"Missing manifest: {MANIFEST_FILE}"
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        assert os.path.exists(ENV_FILE), f"Missing environment: {ENV_FILE}"
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            cls.env = json.load(f)

        assert os.path.exists(STATS_FILE), f"Missing stats JSON: {STATS_FILE}"
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            cls.stats = json.load(f)

    # 1. Manifest existence and valid structure
    def test_01_manifest_exists_and_valid(self):
        self.assertEqual(self.manifest.get("phase"), 14)
        self.assertIn("total_records", self.manifest)
        self.assertIn("algorithms", self.manifest)
        self.assertIn("regimes", self.manifest)
        self.assertIn("repetitions", self.manifest)
        self.assertIn("runs", self.manifest)
        self.assertGreater(len(self.manifest["runs"]), 0)

    # 2. Environment metadata correctness
    def test_02_environment_metadata(self):
        for key in ["os_name", "python_version", "cpu_count_logical", "ram_total_gb", "lb_port", "backend_ports"]:
            self.assertIn(key, self.env)
        self.assertGreater(self.env["cpu_count_logical"], 0)
        self.assertGreater(self.env["ram_total_gb"], 0)

    # 3. Raw dataset exists and schema completeness
    def test_03_raw_dataset_exists_and_schema(self):
        expected_cols = [
            "experiment_id", "run_idx", "algorithm", "regime", "request_id",
            "timestamp", "response_time_ms", "routing_overhead_ms", "success",
            "status_code", "backend_server", "optimal_server", "suboptimal_routing",
            "is_priority_subset",
        ]
        for col in expected_cols:
            self.assertIn(col, self.df_raw.columns)

    # 4. Non-empty observation count
    def test_04_raw_dataset_non_empty(self):
        self.assertGreaterEqual(len(self.df_raw), 500)
        self.assertGreaterEqual(len(self.core_df), 450)

    # 5. All 10 algorithms present in core dataset
    def test_05_all_ten_algorithms_present(self):
        present = set(self.core_df["algorithm"].unique())
        for alg in EXPECTED_ALGORITHMS:
            self.assertIn(alg, present, f"Algorithm {alg} missing from core benchmark dataset")

    # 6. All 9 operational regimes present
    def test_06_all_nine_regimes_present(self):
        present = set(self.core_df["regime"].unique())
        for reg in EXPECTED_REGIMES:
            self.assertIn(reg, present, f"Regime {reg} missing from core benchmark dataset")

    # 7. Matched repetition counts (>= 5) per algorithm x regime
    def test_07_matched_repetitions_count(self):
        grouped = self.core_df.groupby(["algorithm", "regime"])["run_idx"].nunique()
        for count in grouped:
            self.assertGreaterEqual(count, 5, "Fewer than 5 repetitions observed for an algorithm x regime pair")

    # 8. Request ID uniqueness
    def test_08_request_id_uniqueness(self):
        self.assertEqual(len(self.df_raw), self.df_raw["request_id"].nunique(), "Duplicate request_id detected")

    # 9. Latency positivity and finiteness
    def test_09_latency_positivity(self):
        successful = self.df_raw[self.df_raw["success"] == True]
        self.assertTrue((successful["response_time_ms"] > 0.0).all(), "Non-positive latency found")
        self.assertFalse(successful["response_time_ms"].isna().any(), "NaN latency found")
        self.assertFalse(np.isinf(successful["response_time_ms"]).any(), "Infinite latency found")

    # 10. HTTP status code validity
    def test_10_status_code_validity(self):
        codes = self.df_raw["status_code"].unique()
        for c in codes:
            self.assertIn(c, [200, 201, 204, 500, 502, 503, 504], f"Unexpected status code {c}")

    # 11. Summary dataset existence and schema
    def test_11_summary_dataset_exists_and_schema(self):
        expected = [
            "algorithm", "regime", "total_requests", "latency_mean_ms",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
            "routing_overhead_mean_ms", "throughput_completed_rps",
            "routing_accuracy_pct", "suboptimal_routing_rate_pct", "jains_fairness_index",
        ]
        for col in expected:
            self.assertIn(col, self.df_summary.columns)

    # 12. Monotonicity of latency percentiles
    def test_12_latency_monotonicity(self):
        for _, row in self.df_summary.iterrows():
            self.assertLessEqual(row["latency_p50_ms"], row["latency_p95_ms"] + 0.01)
            self.assertLessEqual(row["latency_p95_ms"], row["latency_p99_ms"] + 0.01)
            self.assertLessEqual(row["latency_p99_ms"], row["latency_max_ms"] + 0.01)

    # 13. Error rate bounds [0, 100]
    def test_13_error_rate_bounded(self):
        self.assertTrue((self.df_summary["error_rate_pct"] >= 0.0).all())
        self.assertTrue((self.df_summary["error_rate_pct"] <= 100.0).all())

    # 14. Statistical analysis JSON validity
    def test_14_statistical_analysis_json_valid(self):
        self.assertIn("pairwise_comparisons", self.stats)
        self.assertGreater(len(self.stats["pairwise_comparisons"]), 0)

    # 15. Paired comparison p-values bounded in [0, 1]
    def test_15_paired_comparisons_pvalues(self):
        for k, v in self.stats["pairwise_comparisons"].items():
            self.assertGreaterEqual(v["p_value"], 0.0)
            self.assertLessEqual(v["p_value"], 1.0)
            self.assertIn("cohens_d", v)

    # 16. 95% Confidence Interval structural consistency
    def test_16_confidence_intervals_consistency(self):
        for k, v in self.stats["pairwise_comparisons"].items():
            self.assertLessEqual(v["ci_95_lower"], v["mean_difference_ms"] + 0.01)
            self.assertGreaterEqual(v["ci_95_upper"], v["mean_difference_ms"] - 0.01)

    # 17. Suboptimal routing rate bounds [0, 100]
    def test_17_suboptimal_routing_rate_bounded(self):
        self.assertTrue((self.df_summary["suboptimal_routing_rate_pct"] >= 0.0).all())
        self.assertTrue((self.df_summary["suboptimal_routing_rate_pct"] <= 100.0).all())

    # 18. Jain's Fairness Index bounds [0, 1]
    def test_18_jains_fairness_index_bounded(self):
        self.assertTrue((self.df_summary["jains_fairness_index"] >= 0.0).all())
        self.assertTrue((self.df_summary["jains_fairness_index"] <= 1.01).all())

    # 19. Routing overhead neutral measurement (non-negative, present, finite across all algorithms)
    def test_19_routing_overhead_neutral_measurement(self):
        """Validates that routing overhead is measured correctly, non-negative, finite,
        and present for every algorithm without imposing any artificial ordering assumptions.
        """
        for alg in EXPECTED_ALGORITHMS:
            alg_sub = self.core_df[self.core_df["algorithm"] == alg]
            self.assertFalse(alg_sub.empty, f"Algorithm {alg} has no overhead data")
            self.assertTrue(
                (alg_sub["routing_overhead_ms"] >= 0.0).all(),
                f"Negative routing overhead detected for {alg}",
            )
            self.assertFalse(
                alg_sub["routing_overhead_ms"].isna().any(),
                f"NaN routing overhead detected for {alg}",
            )
            self.assertFalse(
                np.isinf(alg_sub["routing_overhead_ms"]).any(),
                f"Infinite routing overhead detected for {alg}",
            )

    # 20. Fallback rate tracking correctness
    def test_20_fallback_rate_tracking(self):
        self.assertTrue((self.df_summary["fallback_count"] >= 0).all())
        self.assertTrue((self.df_summary["fallback_rate_pct"] >= 0.0).all())

    # 21. Priority evaluation isolation and validity
    def test_21_priority_evaluation_isolation(self):
        self.assertGreater(len(self.prio_df), 0, "Priority evaluation subset should be populated")
        prio_levels = set(self.prio_df["priority"].dropna().unique())
        self.assertTrue(prio_levels.issubset({"LOW", "NORMAL", "HIGH", "CRITICAL"}))
        self.assertTrue(self.prio_df["deadline_ms"].notna().all())

    # 22. Complete presence of all 15 generated figures
    def test_22_all_fifteen_plots_exist(self):
        for plot_name in EXPECTED_PLOTS:
            plot_path = os.path.join(PLOTS_DIR, plot_name)
            self.assertTrue(os.path.exists(plot_path), f"Plot {plot_name} missing from {PLOTS_DIR}")

    # 23. Image files non-empty and valid PNG headers
    def test_23_plots_non_empty_and_valid(self):
        png_magic = b"\x89PNG\r\n\x1a\n"
        for plot_name in EXPECTED_PLOTS:
            plot_path = os.path.join(PLOTS_DIR, plot_name)
            size = os.path.getsize(plot_path)
            self.assertGreater(size, 1000, f"Plot {plot_name} is too small ({size} bytes)")
            with open(plot_path, "rb") as f:
                header = f.read(8)
                self.assertEqual(header, png_magic, f"Plot {plot_name} does not have valid PNG header")

    # 24. Historical dataset immutability
    def test_24_historical_data_immutability(self):
        historical_paths = [
            os.path.join(PROJECT_ROOT, "data", "raw"),
            os.path.join(PROJECT_ROOT, "data", "phase9", "raw_results.csv"),
            os.path.join(PROJECT_ROOT, "data", "phase10", "raw_results.csv"),
            os.path.join(PROJECT_ROOT, "data", "phase12", "phase12_raw.csv"),
            os.path.join(PROJECT_ROOT, "data", "phase13", "meta_dataset.csv"),
        ]
        for p in historical_paths:
            self.assertTrue(os.path.exists(p), f"Historical path {p} was deleted or moved")

    # 25. Candidate model artifacts integrity in models/
    def test_25_candidate_models_integrity(self):
        models = [
            "logistic_regression.joblib",
            "random_forest.joblib",
            "decision_tree.joblib",
            "svm.joblib",
            "xgboost.joblib",
        ]
        for m in models:
            path = os.path.join(MODELS_DIR, m)
            self.assertTrue(os.path.exists(path), f"Candidate model {m} missing")
            obj = joblib.load(path)
            self.assertTrue(hasattr(obj, "predict"), f"Loaded object {m} has no predict method")

    # 26. Optimal server label validity
    def test_26_optimal_server_label_validity(self):
        valid_labels = {"server-1", "server-2", "server-3"}
        observed = set(self.df_raw["optimal_server"].dropna().unique())
        self.assertTrue(observed.issubset(valid_labels), f"Invalid optimal servers: {observed - valid_labels}")


if __name__ == "__main__":
    unittest.main()

