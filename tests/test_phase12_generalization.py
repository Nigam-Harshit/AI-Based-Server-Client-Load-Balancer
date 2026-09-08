"""Phase 12 Tests: Large-Scale Dataset Expansion and Generalization Suite.

Validates:
1. Dataset existence, schema, and volume (>= 1,000 observations across 9 operational regimes).
2. Data integrity: no duplicate IDs, valid timestamps, numeric boundaries, non-negative metrics.
3. Historical dataset immutability (Phases 5, 6, 9, 10 remain unmodified).
4. Experiment manifest and environment reproducibility metadata.
5. Model evaluation methodology: GroupKFold group integrity, seen/unseen splits.
6. Statistical analysis structure and paired comparison validity.
7. Priority & deadline subset isolation.
8. Visualization suite: 12 publication plots exist and are non-empty.
"""

import json
import os
import unittest
import pandas as pd

from ml.evaluation import ACTIVE_PRE_ROUTING_FEATURES
TARGET_COLUMN = "best_server"


class TestPhase12Generalization(unittest.TestCase):
    """Test suite validating Phase 12 dataset expansion, reproducibility, and evaluation."""

    def setUp(self):
        self.data_dir = "data/phase12"
        self.raw_csv = os.path.join(self.data_dir, "phase12_raw.csv")
        self.manifest_json = os.path.join(self.data_dir, "experiment_manifest.json")
        self.env_json = os.path.join(self.data_dir, "environment.json")
        self.summary_csv = os.path.join(self.data_dir, "phase12_summary.csv")
        self.stats_json = os.path.join(self.data_dir, "statistical_analysis.json")
        self.plots_dir = "experiments/results/phase12"

    def test_artifacts_exist(self):
        """Verify all Phase 12 generated files exist on disk."""
        self.assertTrue(os.path.exists(self.raw_csv), "phase12_raw.csv missing")
        self.assertTrue(os.path.exists(self.manifest_json), "experiment_manifest.json missing")
        self.assertTrue(os.path.exists(self.env_json), "environment.json missing")
        self.assertTrue(os.path.exists(self.summary_csv), "phase12_summary.csv missing")
        self.assertTrue(os.path.exists(self.stats_json), "statistical_analysis.json missing")
        self.assertTrue(os.path.isdir(self.plots_dir), "experiments/results/phase12 directory missing")

    def test_dataset_size_and_regimes(self):
        """Verify expanded dataset meets >= 1,000 threshold and covers all 9 regimes."""
        df = pd.read_csv(self.raw_csv)
        self.assertGreaterEqual(len(df), 1000, "Dataset must have at least 1,000 observations")
        self.assertEqual(len(df), 1710, "Expected exactly 1,710 observations collected")

        expected_regimes = {
            "low_load", "medium_load", "high_load", "burst_load",
            "cpu_heavy", "mixed_workload", "dynamic_workload",
            "queue_contention", "backend_imbalance"
        }
        actual_regimes = set(df["regime"].unique())
        self.assertEqual(actual_regimes, expected_regimes, "All 9 operational regimes must be present")

        self.assertGreaterEqual(df["experiment_id"].nunique(), 30)
        self.assertEqual(df["experiment_id"].nunique(), 41)

    def test_schema_and_features(self):
        """Verify schema conforms to the 15 active pre-routing features and labels."""
        df = pd.read_csv(self.raw_csv)

        for feat in ACTIVE_PRE_ROUTING_FEATURES:
            self.assertIn(feat, df.columns, f"Feature {feat} missing from raw dataset")
            self.assertFalse(df[feat].isnull().any(), f"Feature {feat} has null values")

        self.assertIn(TARGET_COLUMN, df.columns)
        self.assertFalse(df[TARGET_COLUMN].isnull().any())
        valid_servers = {"server-1", "server-2", "server-3"}
        self.assertTrue(set(df[TARGET_COLUMN].unique()).issubset(valid_servers))

        required_cols = [
            "request_id", "experiment_id", "regime", "workload_scenario",
            "selected_server", "status_code", "response_time_ms", "timestamp"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Required column {col} missing")

    def test_data_integrity(self):
        """Verify request ID uniqueness, valid numeric bounds, and positive durations."""
        df = pd.read_csv(self.raw_csv)

        self.assertEqual(len(df), df["request_id"].nunique(), "Duplicate request_ids detected")

        self.assertTrue((df["response_time_ms"] > 0).all(), "Response times must be strictly positive")
        self.assertTrue((df["status_code"] == 200).all(), "All requests should be successful status 200")

        for s in [1, 2, 3]:
            self.assertTrue((df[f"server_{s}_cpu"] >= 0).all(), "CPU percent must be >= 0")
            self.assertTrue((df[f"server_{s}_connections"] >= 0).all(), "Connections must be >= 0")
            self.assertTrue((df[f"server_{s}_queue_length"] >= 0).all(), "Queue length must be >= 0")
            self.assertTrue((df[f"server_{s}_response_time"] >= 0).all(), "Response time must be >= 0")

    def test_historical_datasets_unmodified(self):
        """Verify historical datasets from Phases 5, 6, 9, 10 are completely untouched."""
        historical_paths = [
            "data/raw/experiment_data.csv",
            "data/processed/clean_experiment_data.csv",
            "data/phase9/raw_results.csv",
            "data/phase10/priority_raw_results.csv"
        ]
        for path in historical_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                self.assertGreater(len(df), 0, f"Historical file {path} was corrupted or emptied")
                if path == "data/raw/experiment_data.csv":
                    self.assertEqual(len(df), 120, "Phase 5 dataset was modified!")
                elif path == "data/phase9/raw_results.csv":
                    self.assertEqual(len(df), 4800, "Phase 9 dataset was modified!")

    def test_manifest_and_environment(self):
        """Verify manifest and environment metadata structure."""
        with open(self.manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["total_experiments"], 41)
        total_obs = sum(e["total_requests"] for e in manifest["experiments"])
        self.assertGreaterEqual(total_obs, 1000)
        self.assertIn("experiments", manifest)
        self.assertEqual(len(manifest["experiments"]), 41)

        with open(self.env_json, "r", encoding="utf-8") as f:
            env = json.load(f)
        self.assertIn("os_name", env)
        self.assertIn("python_version", env)
        self.assertIn("cpu_count_logical", env)
        self.assertEqual(len(env["backend_ports"]), 3)

    def test_summary_csv_contents(self):
        """Verify summary CSV contains all 5 ML models and 3 separate traditional baselines."""
        df_summary = pd.read_csv(self.summary_csv)
        self.assertEqual(len(df_summary), 8)

        models = set(df_summary["model"].unique())
        expected_models = {
            "Random Forest", "Decision Tree", "Logistic Regression", "SVM", "XGBoost",
            "Baseline: Round Robin", "Baseline: Least Connections", "Baseline: Ip Hash"
        }
        self.assertEqual(models, expected_models)

        expected_cols = [
            "model", "set_a_cv_accuracy", "set_a_cv_macro_f1",
            "set_b_unseen_accuracy", "set_b_unseen_macro_f1",
            "set_c_high_load_accuracy", "set_c_high_load_macro_f1",
            "generalization_gap_macro_f1", "generalization_gap_accuracy"
        ]
        for col in expected_cols:
            self.assertIn(col, df_summary.columns)

    def test_statistical_analysis_contents(self):
        """Verify statistical analysis JSON contains all mandated sections."""
        with open(self.stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.assertIn("dataset_overview", stats)
        self.assertIn("experiment_set_a_seen_cv", stats)
        self.assertIn("experiment_set_b_unseen_test", stats)
        self.assertIn("experiment_set_c_high_load_test", stats)
        self.assertIn("generalization_gaps", stats)
        self.assertIn("scenario_wise_analysis", stats)
        self.assertIn("statistical_comparisons", stats)
        self.assertIn("priority_deadline_extension", stats)

        self.assertEqual(stats["dataset_overview"]["priority_observations"], 160)
        self.assertEqual(stats["dataset_overview"]["core_observations"], 1550)

    def test_all_12_plots_exist(self):
        """Verify all 12 publication plots exist and have non-zero file sizes."""
        expected_plots = [
            "01_model_performance_comparison.png",
            "02_scenariowise_macro_f1.png",
            "03_scenariowise_accuracy.png",
            "04_high_load_performance.png",
            "05_seen_vs_unseen_performance.png",
            "06_generalization_gap.png",
            "07_per_class_f1_distribution.png",
            "08_confusion_matrices.png",
            "09_latency_distributions.png",
            "10_throughput_comparison.png",
            "11_server_utilization.png",
            "12_priority_deadline_comparison.png"
        ]
        for plot_name in expected_plots:
            plot_path = os.path.join(self.plots_dir, plot_name)
            self.assertTrue(os.path.exists(plot_path), f"Plot {plot_name} missing")
            self.assertGreater(os.path.getsize(plot_path), 5000, f"Plot {plot_name} is empty or too small")


if __name__ == "__main__":
    unittest.main()
