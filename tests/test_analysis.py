"""Tests for Phase 6 Dataset Analysis module."""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from experiments.analysis import (
    DatasetAnalyzer,
    FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
)


class TestDatasetAnalysis(unittest.TestCase):
    """Test suite for DatasetAnalyzer functionality."""

    @classmethod
    def setUpClass(cls):
        """Create a representative mock dataset for rigorous unit testing."""
        cls.test_dir = tempfile.mkdtemp()

        # Build mock dataset with 30 observations across 2 scenarios
        rows = []
        timestamps = [1000.0 + i * 0.1 for i in range(30)]
        for i in range(30):
            scenario = "burst_traffic" if i < 15 else "cpu_heavy"
            start_t = timestamps[i] + 0.001
            end_t = start_t + (0.05 if scenario == "burst_traffic" else 0.15)
            row = {
                "experiment_id": f"exp_mock_{1 if i < 15 else 2}",
                "timestamp": timestamps[i],
                "request_id": i + 1,
                "workload_scenario": scenario,
                "request_type": "process",
                "request_size": 100,
                "concurrency": 2 if scenario == "burst_traffic" else 1,
                "request_rate": 10.0 if scenario == "burst_traffic" else None,
                "routing_algorithm": "round_robin",
                # Server 1
                "server_1_cpu": 5.0 if scenario == "burst_traffic" else 20.0,
                "server_1_memory": 0.20,
                "server_1_connections": 1 if i % 2 == 0 else 0,
                "server_1_response_time": 30.0,
                "server_1_network_latency": 10.0,
                "server_1_queue_length": 0,
                # Server 2
                "server_2_cpu": 3.0 if scenario == "burst_traffic" else 15.0,
                "server_2_memory": 0.20,
                "server_2_connections": 0,
                "server_2_response_time": 25.0,
                "server_2_network_latency": 8.0,
                "server_2_queue_length": 0,
                # Server 3
                "server_3_cpu": 4.0 if scenario == "burst_traffic" else 18.0,
                "server_3_memory": 0.20,
                "server_3_connections": 1 if i % 3 == 0 else 0,
                "server_3_response_time": 35.0,
                "server_3_network_latency": 12.0,
                "server_3_queue_length": 0,
                # Outcome
                "selected_server": "http://127.0.0.1:8001" if i % 3 == 0 else ("http://127.0.0.1:8002" if i % 3 == 1 else "http://127.0.0.1:8003"),
                "actual_response_time": 50.0 if scenario == "burst_traffic" else 150.0,
                "request_success": True,
                "best_server": "server-2" if i % 2 == 0 else "server-1",
                "request_start": start_t,
                "request_end": end_t,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        cls.mock_csv_path = os.path.join(cls.test_dir, "processed_mock.csv")
        df.to_csv(cls.mock_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_load_data(self):
        """Test dataset loading and fallback behavior."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        self.assertFalse(analyzer.df.empty)
        self.assertEqual(len(analyzer.df), 30)

        # Test empty dir fallback
        empty_dir = tempfile.mkdtemp()
        try:
            empty_analyzer = DatasetAnalyzer(data_dir=empty_dir, file_pattern="*.csv")
            # If fallback data/raw exists, it loads fallback, else empty
            self.assertIsInstance(empty_analyzer.df, pd.DataFrame)
        finally:
            shutil.rmtree(empty_dir)

    def test_inspect_dataset(self):
        """Test inspection metrics, schema checks, and error rates."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        insp = analyzer.inspect_dataset()

        self.assertEqual(insp["total_observations"], 30)
        self.assertEqual(insp["total_experiments"], 2)
        self.assertTrue(insp["schema_compliant"])
        self.assertEqual(insp["duplicate_records"], 0)
        self.assertEqual(insp["failed_requests"], 0)
        self.assertEqual(len(insp["feature_missing_values"]), 0)
        self.assertEqual(len(insp["outcome_missing_values"]), 0)
        self.assertIn("burst_traffic", insp["scenarios"])
        self.assertIn("cpu_heavy", insp["scenarios"])

    def test_feature_statistics(self):
        """Test 18-feature descriptive statistics computation."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        stats = analyzer.compute_feature_statistics()

        self.assertEqual(len(stats), 18)
        self.assertIn("server_1_cpu", stats.index)
        self.assertIn("server_1_queue_length", stats.index)

        # Queue length should be identified as zero-variance
        self.assertTrue(stats.loc["server_1_queue_length", "is_zero_variance"])
        # CPU should have positive variance
        self.assertFalse(stats.loc["server_1_cpu", "is_zero_variance"])
        self.assertGreater(stats.loc["server_1_cpu", "variance"], 0.0)

    def test_label_analysis(self):
        """Test target class balance and baseline routing optimality."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        lbl = analyzer.analyze_labels()

        self.assertIn("class_counts", lbl)
        self.assertIn("baseline_optimality_rate", lbl)
        self.assertIn("imbalance_ratio", lbl)
        self.assertTrue(lbl["is_balanced"])
        self.assertGreaterEqual(lbl["baseline_optimality_rate"], 0.0)
        self.assertLessEqual(lbl["baseline_optimality_rate"], 1.0)

    def test_compute_correlations(self):
        """Test Pearson & Spearman correlation calculation and zero-variance screening."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        corr = analyzer.compute_correlations()

        self.assertIn("pearson", corr)
        self.assertIn("spearman", corr)
        self.assertIn("zero_variance_features", corr)

        # Zero-variance features should be flagged
        self.assertIn("server_1_queue_length", corr["zero_variance_features"])
        # Target correlation should be populated
        self.assertIn("target_correlation", corr)
        self.assertIsInstance(corr["target_correlation"], dict)

    def test_audit_data_leakage(self):
        """Test causality verification and feature-outcome separation."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        audit = analyzer.audit_data_leakage()

        self.assertTrue(audit["leakage_free"])
        self.assertEqual(len(audit["temporal_violations"]), 0)
        self.assertEqual(len(audit["feature_outcome_overlap"]), 0)
        self.assertEqual(audit["invalid_durations"], 0)

    def test_temporal_properties(self):
        """Test sequential autocorrelation calculation and split strategy."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        temporal = analyzer.analyze_temporal_properties()

        self.assertIn("overall_lag1_autocorrelation", temporal)
        self.assertIn("recommended_split_strategy", temporal)
        self.assertIn("GroupKFold", temporal["recommended_split_strategy"])

    def test_scenario_differentiation(self):
        """Test scenario-grouped summary aggregation."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        sc_df = analyzer.analyze_scenario_differentiation()

        self.assertEqual(len(sc_df), 2)
        self.assertIn("burst_traffic", sc_df.index)
        self.assertIn("cpu_heavy", sc_df.index)
        # CPU-heavy response time should be greater than burst
        self.assertGreater(sc_df.loc["cpu_heavy", "avg_response_ms"], sc_df.loc["burst_traffic", "avg_response_ms"])

    def test_generate_visualizations(self):
        """Test generation of all 5 PNG visualization charts."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        out_vis = tempfile.mkdtemp()
        try:
            plots = analyzer.generate_visualizations(output_dir=out_vis)
            self.assertEqual(len(plots), 5)
            for p in plots:
                self.assertTrue(os.path.exists(p))
                self.assertGreater(os.path.getsize(p), 1000)  # Verify non-empty file
        finally:
            shutil.rmtree(out_vis)

    def test_generate_report_markdown(self):
        """Test comprehensive markdown report compilation."""
        analyzer = DatasetAnalyzer(data_dir=self.test_dir, file_pattern="*.csv")
        out_md_dir = tempfile.mkdtemp()
        out_md = os.path.join(out_md_dir, "test_report.md")
        try:
            content = analyzer.generate_report_markdown(output_path=out_md)
            self.assertTrue(os.path.exists(out_md))
            self.assertIn("Phase 6 — Experimental Dataset Analysis", content)
            self.assertIn("VERDICT: READY FOR ML EXPERIMENTATION", content)
            self.assertIn("Zero Data Leakage", content)
            self.assertIn("Workload Scenario Behavioral Differentiation", content)
        finally:
            shutil.rmtree(out_md_dir)


if __name__ == "__main__":
    unittest.main()
