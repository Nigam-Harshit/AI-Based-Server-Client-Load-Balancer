# Phase 7 — Machine Learning Baselines & Model Comparison Report

**Project:** AI-Based Server-Client Load Balancer using a Random Forest Classifier  
**Phase:** Phase 7 — ML Baselines & Model Comparison  
**Validation Scheme:** GroupKFold Cross-Validation by `experiment_id` (6 folds, Leave-One-Group-Out)  
**Random Seed:** 42  
**Status:** Model Evaluation & Empirical Baseline Comparison Complete

---

## 1. Executive Summary & Verdict

This document provides an objective empirical comparison of multiple machine learning classifiers against the traditional heuristic routing baseline (`round_robin`, `least_connections`, `ip_hash`) on the 120-observation experimental dataset.

### Core Empirical Takeaways:

1. **Baseline Outperformed**: The traditional load balancer achieved only **31.7% accuracy** and **27.7% Macro F1**. Every tested ML model surpassed the traditional baseline in predictive accuracy.
2. **Best-Performing Model: Logistic Regression**:
   - **Macro F1:** **0.6824 ± 0.1302**
   - **Accuracy:** **0.7561 ± 0.1684**
   - **Relative Gain over Baseline:** **+43.9 percentage points** in accuracy and **+40.6 percentage points** in Macro F1.
3. **Random Forest Performance Analysis**:
   - Random Forest achieved **0.5211 ± 0.2246 accuracy** and **0.3091 ± 0.1348 Macro F1**.
   - **Empirical Observation**: Random Forest did *not* outperform Logistic Regression on this dataset. Because the underlying cost objective is primarily a linear combination of normalized queue and latency metrics, regularized linear models (Logistic Regression) generalized superiorly across unseen scenarios with small sample sizes ($N=120$), while decision trees and forests experienced variance across disparate experimental regimes.
4. **Causality & Leakage Verification**: All models strictly evaluated on pre-routing metrics only; preprocessing (standard scaling) was strictly encapsulated within pipelines fit on training folds.

---

## 2. Dataset & Feature Selection

- **Total Observations ($N$):** 120
- **Number of Classes ($K$):** 3 (`server-1`, `server-2`, `server-3`)
- **Total Candidate Features:** 18
- **Features Retained for ML:** 15
- **Features Excluded (Zero-Variance):** `server_1_queue_length`, `server_2_queue_length`, `server_3_queue_length`
- **Post-Routing Outcome Columns Strictly Excluded:** `selected_server`, `actual_response_time`, `request_success`, `request_start`, `request_end`

### Active Pre-Routing Features ($X$):

| Index | Feature Name | Server | Metric Category | Preprocessing Treatment |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `server_1_cpu` | Server 1 | `cpu` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 2 | `server_1_memory` | Server 1 | `memory` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 3 | `server_1_connections` | Server 1 | `connections` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 4 | `server_1_response_time` | Server 1 | `response_time` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 5 | `server_1_network_latency` | Server 1 | `network_latency` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 6 | `server_2_cpu` | Server 2 | `cpu` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 7 | `server_2_memory` | Server 2 | `memory` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 8 | `server_2_connections` | Server 2 | `connections` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 9 | `server_2_response_time` | Server 2 | `response_time` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 10 | `server_2_network_latency` | Server 2 | `network_latency` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 11 | `server_3_cpu` | Server 3 | `cpu` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 12 | `server_3_memory` | Server 3 | `memory` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 13 | `server_3_connections` | Server 3 | `connections` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 14 | `server_3_response_time` | Server 3 | `response_time` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |
| 15 | `server_3_network_latency` | Server 3 | `network_latency` | StandardScaler (in Pipeline for LR/SVM; raw for Trees) |

---

## 3. Validation Methodology & Leakage Prevention

### Validation Strategy: `GroupKFold(n_splits=6, groups=experiment_id)`
Naive random train/test splitting was strictly avoided. In a sequential load balancing trace, consecutive requests within the same experimental run exhibit temporal serial correlation (lag-1 autocorrelation). Shuffling requests across folds would cause intra-stream data leakage.

By grouping by `experiment_id`, **every fold holds out an entire unseen experimental workload run** (e.g. holding out the `mixed` experiment while training on `burst_traffic`, `dynamic`, `cpu_heavy`, `low_traffic`, `medium_traffic`). This evaluates true out-of-distribution generalization.

### Pipeline Preprocessing:
- For **Logistic Regression** and **SVM**, `StandardScaler` is wrapped inside a `sklearn.pipeline.Pipeline`.
- The mean and variance for scaling are fit strictly on $X_{\text{train}}$ of each fold, never touching $X_{\text{val}}$.
- For **Random Forest**, **Decision Tree**, and **XGBoost**, raw values are passed directly into tree splitting criteria.

---

## 4. Consolidated Model Comparison

Evaluation metrics across all 6 validation folds (Mean ± Standard Deviation):

| Model | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Traditional Baseline** | 0.3167 | 0.2767 | 0.2976 | 0.2859 | 0.2851 |
| **Random Forest** | 0.5211 ± 0.2246 | **0.3091 ± 0.1348** | 0.4330 ± 0.2445 | 0.2914 ± 0.1274 | 0.3847 ± 0.1203 |
| **Decision Tree** | 0.5006 ± 0.2071 | **0.3537 ± 0.1306** | 0.4687 ± 0.1975 | 0.3652 ± 0.1262 | 0.4244 ± 0.1622 |
| **Logistic Regression** | 0.7561 ± 0.1684 | **0.6824 ± 0.1302** | 0.7459 ± 0.1733 | 0.7387 ± 0.0986 | 0.6961 ± 0.1457 |
| **SVM** | 0.5489 ± 0.1652 | **0.3655 ± 0.1494** | 0.4777 ± 0.2016 | 0.4252 ± 0.1650 | 0.4330 ± 0.1293 |
| **XGBoost** | 0.5656 ± 0.2102 | **0.4246 ± 0.1986** | 0.5304 ± 0.2335 | 0.4775 ± 0.2715 | 0.4630 ± 0.1784 |

> [!NOTE]
> **Primary Metric: Macro F1**. In multiclass routing across 3 servers, Macro F1 treats all servers equally, preventing high accuracy on a single dominant server from masking poor routing decisions on other nodes.

---

## 5. Baseline Comparison & Headroom Analysis

| Metric | Traditional Baseline | Best ML (Logistic Regression) | Absolute Gain | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 31.7% | 75.6% | +43.9% | +138.8% |
| **Macro F1** | 27.7% | 68.2% | +40.6% | +146.6% |
| **Weighted F1** | 29.8% | 74.6% | +44.8% | — |

### Distinction: Prediction Accuracy vs Actual System Performance
- **Prediction Accuracy** quantifies how frequently the ML classifier designates the exact backend node that minimizes the pre-routing cost proxy function.- **Actual System Performance Improvement** (to be experimentally measured in Phase 8) represents end-to-end response time reductions, p95 latency drops, and server throughput improvements when the ML model directly drives traffic routing decisions in the live load balancer.

---

## 6. Random Forest Detailed Analysis

As the proposed primary algorithm for the project, Random Forest was analyzed in depth:

- **Cross-Validated Accuracy:** 0.5211 ± 0.2246
- **Cross-Validated Macro F1:** 0.3091 ± 0.1348
- **Per-Class F1-Scores:** `server-1`: 0.738, `server-2`: 0.364, `server-3`: 0.098

### Feature Importance Analysis (MDI Gini vs Permutation Importance):

| Rank | Feature | Gini Importance (MDI) | Permutation Importance (Macro F1 Δ) |
| :--- | :--- | :--- | :--- |
| 1 | `server_1_connections` | 0.1303 | 0.1520 ± 0.0340 |
| 2 | `server_1_response_time` | 0.1233 | 0.0251 ± 0.0171 |
| 3 | `server_3_network_latency` | 0.1017 | 0.1319 ± 0.0147 |
| 4 | `server_1_network_latency` | 0.0947 | 0.0422 ± 0.0113 |
| 5 | `server_2_network_latency` | 0.0902 | 0.0688 ± 0.0155 |

> [!CAUTION]
> **Causality Disclaimer**: Feature importance identifies statistical association and split frequency within the decision tree ensemble; it does not constitute causal proof that altering that specific metric alone will dictate latency.

---

## 7. Limitations & Empirical Risks

1. **Sample Size Constraint ($N=120$):** With 120 total samples partitioned into 6 folds (~20 samples per validation fold), standard deviations across folds are moderate ($\pm 0.12$ to $\pm 0.21$). High fold-to-fold variance reflects diverse scenario characteristics.
2. **Loopback Network Latency:** Probes executed on local loopback (`127.0.0.1`) have minimal propagation delay (< 30 ms). Real-world LAN/WAN environments will exhibit higher latency variation.
3. **Absence of Server Queue Saturation:** Queue length was constant (0) in these threaded runs. In heavily saturated production environments with socket backlogs, queue length will play a major predictive role.

---

## 8. Conclusion & Gate Decision

1. **Validation Complete**: Successfully evaluated Random Forest, Decision Tree, Logistic Regression, SVM, and XGBoost using GroupKFold cross-validation.
2. **Superior Baseline Outperformance**: ML routing achieves up to **75.6% accuracy**, more than double the traditional baseline (31.7%).
3. **Model Selection Insight**: While Random Forest delivers solid predictive power (52.1% accuracy, outperforming baseline by +20.6%), regularized linear models (Logistic Regression: 75.6%) provide competitive generalization under small sample regimes.
4. **Recommendation for Phase 8**: Proceed to integrate ML inference into the load balancer runtime, supporting configurable model backends (`RandomForest` and `LogisticRegression`) to evaluate real-time system performance gains.
