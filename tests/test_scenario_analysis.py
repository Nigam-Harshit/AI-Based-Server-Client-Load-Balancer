"""Unit and integration tests for Phase 7.5 Scenario-Wise ML Analysis."""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

from ml.scenario_analysis import ScenarioAnalyzer, SCENARIO_METADATA, LOAD_REGIMES
from ml.evaluation import LABEL_NAMES


class TestScenarioAnalysis(unittest.TestCase):
    """Test suite for scenario-wise model performance and robustness analysis."""

    @classmethod
    def setUpClass(cls):
        """Build representative mock dataset spanning 3 distinct scenarios and experiment groups."""
        cls.test_dir = tempfile.mkdtemp()

        rows = []
        scenarios = ["low_traffic", "burst_traffic", "cpu_heavy"]
        for exp_idx, sc in enumerate(scenarios):
            exp_id = f"exp_{sc}_r{exp_idx+1}"
            for i in range(15):
                req_id = exp_idx * 15 + i + 1
                best_srv = LABEL_NAMES[(i + exp_idx) % 3]
                selected_url = f"http://127.0.0.1:800{(i % 3) + 1}"
                row = {
                    "experiment_id": exp_id,
                    "timestamp": 1000.0 + req_id,
                    "request_id": req_id,
                    "workload_scenario": sc,
                    "request_type": "process",
                    "request_size": 100,
                    "concurrency": 2 if sc == "low_traffic" else (20 if sc == "burst_traffic" else 5),
                    "request_rate": 5.0 if sc == "low_traffic" else None,
                    "routing_algorithm": "round_robin",
                    # Server 1
                    "server_1_cpu": 1.0 if sc == "low_traffic" else (5.0 if sc == "burst_traffic" else 22.0),
                    "server_1_memory": 0.20,
                    "server_1_connections": 0 if sc == "low_traffic" else (i % 2),
                    "server_1_response_time": 25.0 + i,
                    "server_1_network_latency": 10.0 + (i % 3),
                    "server_1_queue_length": 0,
                    # Server 2
                    "server_2_cpu": 1.0 if sc == "low_traffic" else (4.0 if sc == "burst_traffic" else 18.0),
                    "server_2_memory": 0.20,
                    "server_2_connections": 0,
                    "server_2_response_time": 22.0 + i,
                    "server_2_network_latency": 8.0 + (i % 2),
                    "server_2_queue_length": 0,
                    # Server 3
                    "server_3_cpu": 1.0 if sc == "low_traffic" else (3.0 if sc == "burst_traffic" else 20.0),
                    "server_3_memory": 0.20,
                    "server_3_connections": 1 if i % 3 == 0 else 0,
                    "server_3_response_time": 30.0 + i,
                    "server_3_network_latency": 12.0 + (i % 4),
                    "server_3_queue_length": 0,
                    # Outcomes
                    "selected_server": selected_url,
                    "actual_response_time": 30.0 if sc == "low_traffic" else (75.0 if sc == "burst_traffic" else 160.0),
                    "request_success": True,
                    "best_server": best_srv,
                    "request_start": 1000.0 + req_id + 0.001,
                    "request_end": 1000.0 + req_id + 0.05,
                }
                rows.append(row)

        cls.mock_df = pd.DataFrame(rows)
        cls.mock_csv_path = os.path.join(cls.test_dir, "processed_sc_mock.csv")
        cls.mock_df.to_csv(cls.mock_csv_path, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_out_of_fold_predictions_shape(self):
        """Test that out-of-fold predictions match dataset dimensions."""
        analyzer = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        preds = analyzer.compute_out_of_fold_predictions()

        self.assertIn("Traditional Baseline", preds)
        self.assertIn("Random Forest", preds)
        self.assertIn("Logistic Regression", preds)
        self.assertEqual(len(preds["Traditional Baseline"]), len(self.mock_df))
        self.assertEqual(len(preds["Random Forest"]), len(self.mock_df))

    def test_evaluate_scenarios_structure(self):
        """Test that scenario evaluations produce complete metric dictionaries and tables."""
        analyzer = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        results = analyzer.evaluate_scenarios()

        self.assertIn("scenario_metrics", results)
        self.assertIn("f1_table", results)
        self.assertIn("accuracy_table", results)
        self.assertIn("robustness", results)
        self.assertIn("regime_results", results)

        sc_metrics = results["scenario_metrics"]
        for sc in ["low_traffic", "burst_traffic", "cpu_heavy"]:
            self.assertIn(sc, sc_metrics)
            self.assertIn("best_model_macro_f1", sc_metrics[sc])
            self.assertIn("best_model_accuracy", sc_metrics[sc])
            self.assertGreaterEqual(sc_metrics[sc]["best_macro_f1_value"], 0.0)

    def test_robustness_calculation(self):
        """Test robustness metric calculation (mean, std, min, max, range)."""
        analyzer = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        results = analyzer.evaluate_scenarios()
        robustness = results["robustness"]

        for m_name in ["Random Forest", "Logistic Regression"]:
            self.assertIn(m_name, robustness)
            rob = robustness[m_name]
            self.assertIn("mean_macro_f1", rob)
            self.assertIn("std_macro_f1", rob)
            self.assertIn("f1_range", rob)
            self.assertGreaterEqual(rob["max_macro_f1"], rob["min_macro_f1"])
            self.assertAlmostEqual(rob["f1_range"], rob["max_macro_f1"] - rob["min_macro_f1"])

    def test_load_regime_grouping(self):
        """Test load regime grouping aggregation."""
        analyzer = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        results = analyzer.evaluate_scenarios()
        regimes = results["regime_results"]

        self.assertIn("Low Load", regimes)
        self.assertEqual(regimes["Low Load"]["observations"], 15)
        self.assertIn("Logistic Regression", regimes["Low Load"]["models"])

    def test_generate_visualizations(self):
        """Test that all 4 diagnostic plots are generated and non-empty."""
        analyzer = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        results = analyzer.evaluate_scenarios()

        out_vis = tempfile.mkdtemp()
        try:
            plots = analyzer.generate_visualizations(results, output_dir=out_vis)
            self.assertEqual(len(plots), 4)
            for p in plots:
                self.assertTrue(os.path.exists(p))
                self.assertGreater(os.path.getsize(p), 1000)
        finally:
            shutil.rmtree(out_vis)

    def test_generate_report_markdown(self):
        """Test scenario-wise markdown report compilation."""
        analyzer = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        results = analyzer.evaluate_scenarios()

        out_md_dir = tempfile.mkdtemp()
        out_md = os.path.join(out_md_dir, "test_scenario_report.md")
        try:
            report_text = analyzer.generate_report_markdown(results, output_path=out_md)
            self.assertTrue(os.path.exists(out_md))
            self.assertIn("Phase 7.5 — Scenario-Wise Machine Learning", report_text)
            self.assertIn("Answers to Core Research Questions", report_text)
            self.assertIn("Macro F1-Score by Workload Scenario", report_text)
            self.assertIn("Model Robustness & Stability Analysis", report_text)
            self.assertIn("Load-Based Regime Analysis", report_text)
            self.assertIn("Final Verdict & Guidance for Phase 8", report_text)
        finally:
            shutil.rmtree(out_md_dir)

    def test_reproducibility(self):
        """Test that scenario evaluations with identical random seeds yield identical metrics."""
        analyzer1 = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)
        analyzer2 = ScenarioAnalyzer(data_dir=self.test_dir, random_state=42)

        res1 = analyzer1.evaluate_scenarios()
        res2 = analyzer2.evaluate_scenarios()

        df_f1_1 = res1["f1_table"]
        df_f1_2 = res2["f1_table"]

        for col in ["Random Forest", "Logistic Regression"]:
            np.testing.assert_allclose(df_f1_1[col], df_f1_2[col], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
