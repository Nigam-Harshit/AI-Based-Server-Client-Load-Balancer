"""Unit and integration tests for Phase 9 live benchmarking, analysis, and edge suites."""

import json
import os
import unittest
import pandas as pd

from benchmarks.live_comparison import Phase9BenchmarkRunner, ALGORITHMS, SCENARIOS
from benchmarks.edge_tests import EdgeConditionTester
from benchmarks.analysis import Phase9Analyzer


class TestPhase9Benchmarks(unittest.TestCase):
    """Test suite validating Phase 9 benchmark configuration, execution, analysis, and plots."""

    def setUp(self):
        self.output_dir = "data/phase9"
        self.raw_csv = os.path.join(self.output_dir, "raw_results.csv")
        self.summaries_csv = os.path.join(self.output_dir, "run_summaries.csv")
        self.env_json = os.path.join(self.output_dir, "environment.json")
        self.edge_json = os.path.join(self.output_dir, "edge_conditions.json")
        self.stats_json = os.path.join(self.output_dir, "statistical_analysis.json")

    def test_datasets_exist_and_non_empty(self):
        """Verify Phase 9 datasets exist and have expected structure."""
        self.assertTrue(os.path.exists(self.raw_csv), "raw_results.csv missing")
        self.assertTrue(os.path.exists(self.summaries_csv), "run_summaries.csv missing")
        self.assertTrue(os.path.exists(self.env_json), "environment.json missing")
        self.assertTrue(os.path.exists(self.edge_json), "edge_conditions.json missing")
        self.assertTrue(os.path.exists(self.stats_json), "statistical_analysis.json missing")

        df_raw = pd.read_csv(self.raw_csv)
        self.assertGreater(len(df_raw), 0)
        self.assertEqual(len(df_raw), 4800)

        df_summaries = pd.read_csv(self.summaries_csv)
        self.assertEqual(len(df_summaries), 84)  # 7 scenarios * 4 algorithms * 3 reps

    def test_all_scenarios_and_algorithms_covered(self):
        """Verify all algorithms and scenarios are recorded in run_summaries."""
        df_summaries = pd.read_csv(self.summaries_csv)
        algos_in_data = set(df_summaries["algorithm"].unique())
        scenarios_in_data = set(df_summaries["scenario"].unique())

        self.assertEqual(algos_in_data, set(ALGORITHMS))
        self.assertEqual(scenarios_in_data, set(SCENARIOS))

    def test_environment_metadata_structure(self):
        """Verify environment metadata captures reproducibility parameters."""
        with open(self.env_json, "r", encoding="utf-8") as f:
            env = json.load(f)
        self.assertIn("os_name", env)
        self.assertIn("python_version", env)
        self.assertIn("cpu_count_logical", env)
        self.assertIn("model_path", env)
        self.assertEqual(env["backend_count"], 3)
        self.assertEqual(env["repetitions"], 3)

    def test_edge_conditions_data(self):
        """Verify edge conditions data contains failure, recovery, and burst results."""
        with open(self.edge_json, "r", encoding="utf-8") as f:
            edge = json.load(f)
        self.assertIn("failure_and_recovery", edge)
        self.assertIn("short_burst_stress", edge)

        failure_phase = edge["failure_and_recovery"]["failure_phase"]
        self.assertEqual(failure_phase["successful"], 10)

        burst = edge["short_burst_stress"]
        for algo in ALGORITHMS:
            self.assertIn(algo, burst)
            self.assertEqual(burst[algo]["successful"], 50)

    def test_all_10_plots_generated(self):
        """Verify all 10 mandated high-resolution research plots are generated."""
        expected_plots = [
            "01_throughput_by_scenario.png",
            "02_p50_latency.png",
            "03_p95_latency.png",
            "04_p99_latency.png",
            "05_cpu_utilization.png",
            "06_backend_distribution.png",
            "07_ml_confidence_distribution.png",
            "08_ml_fallback_rate.png",
            "09_performance_vs_workload_intensity.png",
            "10_algorithm_robustness.png",
        ]
        plot_dir = "docs/plots"
        for p in expected_plots:
            plot_path = os.path.join(plot_dir, p)
            self.assertTrue(os.path.exists(plot_path), f"Plot missing: {p}")
            self.assertGreater(os.path.getsize(plot_path), 1000, f"Plot {p} is empty or corrupted")

    def test_phase5_dataset_untouched(self):
        """Verify Phase-5 dataset remains pristine and unedited."""
        p5_raw = os.listdir("data/raw")
        p5_proc = os.listdir("data/processed")
        self.assertGreater(len(p5_raw), 0)
        self.assertGreater(len(p5_proc), 0)
        # Check no phase9 files leaked into phase5 directories
        for f in p5_raw + p5_proc:
            self.assertFalse(f.startswith("p9_"), f"Phase 9 file leaked into Phase 5 directory: {f}")


if __name__ == "__main__":
    unittest.main()

