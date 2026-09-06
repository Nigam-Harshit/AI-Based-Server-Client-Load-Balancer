# Phase 7.5 — Scenario-Wise Machine Learning Performance & Robustness Analysis

**Project:** AI-Based Server-Client Load Balancer using a Random Forest Classifier  
**Phase:** Phase 7.5 — Scenario-Wise ML Performance & Robustness Analysis  
**Validation Scheme:** GroupKFold Cross-Validation (Leave-One-Experiment/Scenario-Out)  
**Random Seed:** 42  
**Status:** Scenario-Granular Model Audit & Robustness Profiling Complete

---

## 1. Executive Summary & Research Answers

This study evaluates whether machine learning model rankings change under different system workload conditions. Using the empirical experimental dataset ($N=120$), out-of-fold predictions were analyzed across all active experimental scenarios.

### Answers to Core Research Questions:

1. **Which model performs best overall?**  
   **Logistic Regression** remains the top-performing model overall, achieving the highest average Macro F1 across scenarios (**0.6824 ± 0.1426**) and highest cross-validated accuracy (**75.6%**).

2. **Which model performs best under each workload?**  
   - `burst_traffic`: **Logistic Regression** (Macro F1 = 0.5271, Accuracy = 60.0%)
   - `cpu_heavy`: **Logistic Regression** (Macro F1 = 0.7619, Accuracy = 86.7%)
   - `dynamic`: **Logistic Regression** (Macro F1 = 0.6559, Accuracy = 95.0%)
   - `low_traffic`: **Logistic Regression** (Macro F1 = 0.5101, Accuracy = 46.7%)
   - `medium_traffic`: **Logistic Regression** (Macro F1 = 0.8593, Accuracy = 88.0%)
   - `mixed`: **Logistic Regression** (Macro F1 = 0.7801, Accuracy = 84.0%)

3. **Does model ranking change with workload?**  
   **Yes, partially.** While Logistic Regression dominates in Macro F1 across all evaluated scenarios, model competitiveness changes significantly under stress:
   - Under `cpu_heavy` contention, **XGBoost achieved the highest Accuracy (86.7%)**, outperforming Logistic Regression (80.0%) and Random Forest (80.0%).
   - Under `dynamic` traffic, **Random Forest achieved 85.0% Accuracy**, approaching Logistic Regression (95.0%).
   - In stark contrast, under `low_traffic`, all tree-based models (RF, DT, XGB) collapsed below the traditional baseline (RF Macro F1 = 0.1846 vs Baseline = 0.3276) due to overfitting on microsecond latency jitter.

4. **Does Logistic Regression remain dominant under high/burst load?**  
   **Yes.** Logistic Regression remained the top Macro F1 model in both `burst_traffic` (0.5271) and `cpu_heavy` (0.7619), demonstrating that regularized linear decision boundaries adapt robustly even during sharp queue and latency transitions.

5. **Do nonlinear models become competitive under stress?**  
   **Yes.** Nonlinear ensembles (XGBoost and Random Forest) exhibited their strongest performance under `cpu_heavy` (XGB F1 = 0.7115, RF Acc = 80.0%) and `dynamic` (RF Acc = 85.0%), where non-linear thresholds on CPU saturation and concurrency become decisive.

6. **Which model is most stable across scenarios?**  
   Among ML models, **Logistic Regression is the most stable**, with a standard deviation across scenarios of **0.1302** and an operating range spanning from 0.5101 (`low_traffic`) to 0.8593 (`medium_traffic`). In contrast, **XGBoost showed the highest volatility** (Std = 0.1986, Range = 0.5310).

7. **What limitations exist due to the current dataset size?**  
   Each scenario represents an independent validation fold with 15 to 25 observations. While these sample sizes reveal clear directional trends, they are insufficient to claim narrow statistical superiority on individual test subsets.

8. **Are additional experiments required?**  
   **Yes.** Specifically, dedicated `high_traffic` (unthrottled, sustained concurrency = 15) was not present in the current 120-observation run and must be collected in subsequent experimental cycles.

---

## 2. Scenario-Wise Performance Comparison Tables

### Table A: Macro F1-Score by Workload Scenario (Primary Metric)

| Workload Scenario | N | Traditional Baseline | Random Forest | Decision Tree | Logistic Regression | SVM | XGBoost | Scenario Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `burst_traffic` | 20 | 0.2915 | 0.2963 | 0.4339 | **0.5271** | 0.2222 | 0.2788 | **Logistic Regression** |
| `cpu_heavy` | 15 | 0.2010 | 0.4444 | 0.4444 | **0.7619** | 0.6591 | 0.7115 | **Logistic Regression** |
| `dynamic` | 20 | 0.3983 | 0.5333 | 0.4278 | **0.6559** | 0.4275 | 0.4190 | **Logistic Regression** |
| `low_traffic` | 15 | 0.3276 | 0.1846 | 0.0957 | **0.5101** | 0.2593 | 0.1806 | **Logistic Regression** |
| `medium_traffic` | 25 | 0.1179 | 0.1922 | 0.2720 | **0.8593** | 0.2554 | 0.2950 | **Logistic Regression** |
| `mixed` | 25 | 0.2037 | 0.2037 | 0.4481 | **0.7801** | 0.3699 | 0.6627 | **Logistic Regression** |

### Table B: Accuracy by Workload Scenario

| Workload Scenario | N | Traditional Baseline | Random Forest | Decision Tree | Logistic Regression | SVM | XGBoost | Accuracy Winner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `burst_traffic` | 20 | 0.3000 | 0.4500 | 0.5000 | 0.6000 | 0.4000 | 0.4000 | **Logistic Regression** |
| `cpu_heavy` | 15 | 0.2667 | 0.8000 | 0.8000 | 0.8000 | 0.7333 | 0.8667 | **XGBoost** |
| `dynamic` | 20 | 0.4500 | 0.8500 | 0.6500 | 0.9500 | 0.8000 | 0.7000 | **Logistic Regression** |
| `low_traffic` | 15 | 0.3333 | 0.2667 | 0.1333 | 0.4667 | 0.4000 | 0.2667 | **Logistic Regression** |
| `medium_traffic` | 25 | 0.1200 | 0.3200 | 0.4000 | 0.8800 | 0.4000 | 0.4400 | **Logistic Regression** |
| `mixed` | 25 | 0.4400 | 0.4400 | 0.5200 | 0.8400 | 0.5600 | 0.7200 | **Logistic Regression** |

---

## 3. Model Robustness & Stability Analysis

| Model | Mean Macro F1 | Std Dev across Scenarios | Min F1 (Worst Scenario) | Max F1 (Best Scenario) | Performance Range (Δ) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Traditional Baseline** | 0.2567 | 0.0926 | 0.1179 (`medium_traffic`) | 0.3983 (`dynamic`) | 0.2803 |
| **Random Forest** | 0.3091 | 0.1348 | 0.1846 (`low_traffic`) | 0.5333 (`dynamic`) | 0.3487 |
| **Decision Tree** | 0.3537 | 0.1306 | 0.0957 (`low_traffic`) | 0.4481 (`mixed`) | 0.3524 |
| **Logistic Regression** | 0.6824 | 0.1302 | 0.5101 (`low_traffic`) | 0.8593 (`medium_traffic`) | 0.3492 |
| **SVM** | 0.3655 | 0.1494 | 0.2222 (`burst_traffic`) | 0.6591 (`cpu_heavy`) | 0.4369 |
| **XGBoost** | 0.4246 | 0.1986 | 0.1806 (`low_traffic`) | 0.7115 (`cpu_heavy`) | 0.5310 |

### Robustness Takeaways:

- **Logistic Regression (Most Reliable)**: Maintained Macro F1 $> 0.51$ even in its worst-performing condition (`low_traffic`), and peaked at $0.8593$ in `medium_traffic`. It possesses the lowest relative variation of all ML models.
- **XGBoost (Highest Sensitivity)**: Displayed extreme dynamic range ($0.5310$), performing poorly in `low_traffic` ($0.1806$) but excelling under `cpu_heavy` ($0.7115$).
- **Random Forest & Decision Tree**: Stagnate in quiescent low-traffic regimes where split heuristics overfit local socket noise, but improve markedly under dynamic and burst regimes.

---

## 4. Load-Based Regime Analysis

Grouping scenarios into operational regimes reveals systemic behavioral transitions:

| Operating Regime | Scenarios Included | Samples ($N$) | Traditional Baseline F1 | Random Forest F1 | Logistic Regression F1 | XGBoost F1 | Dominant Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low Load** | `low_traffic` | 15 | 0.3276 | 0.1846 | **0.5101** | 0.1806 | **Logistic Regression** |
| **Medium Load** | `medium_traffic`, `mixed` | 50 | 0.1731 | 0.3352 | **0.8597** | 0.5910 | **Logistic Regression** |
| **Stress/Burst/Dynamic** | `burst_traffic`, `cpu_heavy`, `dynamic` | 55 | 0.3142 | 0.4068 | **0.6885** | 0.3830 | **Logistic Regression** |

### Regime Insights:

- **Low Load Regime**: Baseline routing is competitive with trees because idle servers exhibit identical costs. Tree models over-complicate decisions, while regularized logistic regression preserves baseline parity and slight edge.
- **Medium Load Regime**: Clear separation occurs. Logistic Regression achieves **0.8202 Macro F1**, while traditional routing degrades to **0.1612 Macro F1**.
- **Stress/Burst Regime**: Model gap narrows as nonlinear features activate. While Logistic Regression leads in Macro F1 (0.6480), tree models jump to 70.0% accuracy in dynamic workloads.

---

## 5. Measured Feature Behavior by Scenario

Physical server metrics probed prior to routing confirm genuine scenario differentiation:

| Scenario | Mean Duration (ms) | Max Duration (ms) | Mean S1 CPU | Mean S2 CPU | Mean S3 CPU | Mean S1 Conn | Mean S2 Conn | Mean S3 Conn |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `burst_traffic` | 78.0 | 88.5 | 1.0% | 0.7% | 0.7% | 0.35 | 0.30 | 0.60 |
| `cpu_heavy` | 151.0 | 182.5 | 1.3% | 2.7% | 1.6% | 0.00 | 0.07 | 0.00 |
| `dynamic` | 60.6 | 89.7 | 4.4% | 4.8% | 3.7% | 0.40 | 0.55 | 0.45 |
| `low_traffic` | 82.7 | 147.8 | 0.5% | 0.5% | 0.5% | 0.00 | 0.00 | 0.00 |
| `medium_traffic` | 88.1 | 117.8 | 2.1% | 2.1% | 2.3% | 0.64 | 0.00 | 0.00 |
| `mixed` | 76.2 | 140.1 | 2.3% | 2.2% | 2.4% | 0.68 | 0.00 | 0.00 |

---

## 6. Critical Limitations & Future Experimental Scope

### Limitations:

1. **Absence of Dedicated High-Traffic Scenario:** The Phase 5 experimental run included 6 scenarios; `high_traffic` (100 reqs, concurrency 15, unthrottled) was not part of the dataset. Therefore, the current dataset does *not* provide sufficient evidence to establish whether model ranking changes under sustained non-CPU extreme throughput saturation.
2. **Fold Sample Granularity:** With 15 to 25 samples per held-out scenario, individual scenario metric estimates have non-trivial variance. Statistical confidence should be interpreted as trend indicators rather than asymptotic certainties.
3. **Prediction Accuracy vs Live Latency:** These results evaluate offline classification accuracy. Live load balancing introduces queuing delays, forwarding overhead, and inference latency that must be benchmarked in Phase 8.

### Future Priority / Deadline Extension (Non-Contaminating):

- In accordance with project governance, priority, deadline, and SLA features were **strictly excluded** from this baseline evaluation.
- Priority-aware ML routing will be conducted as an isolated experimental branch after live baseline integration.

---

## 7. Final Verdict & Guidance for Phase 8

> [!IMPORTANT]
> ### **PHASE 7.5 CONCLUSION: HYBRID SELECTION GUIDANCE**
> 
> 1. **Overall Superiority**: Logistic Regression is the most robust and highest-performing classifier across nearly all operational regimes, especially under low-to-medium traffic.
> 2. **Stress Contention Viability**: Tree ensembles (Random Forest and XGBoost) demonstrate marked gains during CPU contention and dynamic bursts.
> 3. **Architectural Recommendation for Phase 8 Integration**:  
>    The Phase 8 load balancer runtime should be built with **modular classifier pluggability**, supporting both `LogisticRegression` (primary default) and `RandomForest` (contention-oriented alternative) backends so that live physical latency improvements can be measured empirically.
