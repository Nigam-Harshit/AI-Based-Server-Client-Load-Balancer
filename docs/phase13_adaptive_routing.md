# Phase 13: Adaptive / Context-Aware ML Routing Analysis

## 1. Motivation

In Phase 12, large-scale dataset expansion ($N=1,710$) demonstrated that **no single machine-learning model universally dominated across all operational conditions**:
* **SVM** achieved the highest cross-validation score on seen configurations (Accuracy: 96.65%, Macro F1: 92.39%).
* **Tree-based ensembles (Random Forest, XGBoost)** and Decision Trees exhibited superior generalization to out-of-distribution unseen configurations (Accuracy: 94.51%, Macro F1: 89.14%, with a negative generalization gap of $-3.26\%$).
* **Logistic Regression** degraded sharply under distribution shifts (Macro F1 dropped to 71.85%, with a positive generalization gap of $+14.03\%$).
* Under heavy CPU saturation and queue contention, tree-based models provided robust, deterministic partitions.

This empirical divergence motivated the central hypothesis of Phase 13:
> **Can the load balancer dynamically select the most appropriate routing model based on the current observed system/workload context, instead of permanently using one globally selected model?**

---

## 2. Core Research Verdict: Fixed Tree Ensembles Outperform Adaptive Routing

In accordance with scientific rigor, Phase 13 was designed to evaluate adaptive routing without assuming its superiority. The experiment tested whether adaptive routing:
1. Clearly outperforms fixed models (Outcome A),
2. Performs similarly to the best fixed model (Outcome B),
3. Improves classification accuracy but introduces excessive latency (Outcome C), or
4. Suffers from switching instability and performs worse (Outcome D).

### Empirical Verdict: **OUTCOME B / C (Fixed Models are Superior)**

The empirical results from 5-fold `GroupKFold` cross-validation across 1,550 core observations and 37 experiments demonstrate that **context-aware model selection does NOT provide an advantage over using a single robust fixed model (Random Forest or XGBoost)**:
1. **Routing Effectiveness**: Fixed **Random Forest, Decision Tree, and XGBoost** achieved the highest cross-validation routing accuracy (**93.44%**) and Macro F1 (**92.94%**), with the lowest suboptimal routing rate (**6.56%**). In comparison, both **Adaptive Policy Selector** and **Adaptive Meta-Selector** achieved lower accuracy (**92.65%**) and lower Macro F1 (**89.53%**), with a suboptimal routing rate of **7.35%**.
2. **Model Selection Accuracy vs. Routing Effectiveness**: The adaptive selectors correctly identified an accurate candidate model in **92.65%** of cases. However, correctly predicting the model that won historically on training data did not translate to superior routing effectiveness on held-out test experiments. In fact, routing effectiveness was bounded by the candidate models themselves.
3. **Overhead & Latency Penalty**:
   - Fixed models exhibited sub-millisecond inference times: **0.004 ms – 0.019 ms**.
   - Adaptive Policy Selector added **0.188 ms** of decision overhead.
   - Adaptive Meta-Selector introduced **1.916 ms** of overhead (~100x slower decision latency than fixed models) due to dual feature evaluation and meta-model inference.
4. **Statistical Significance**: A matched paired t-test between Adaptive Policy Routing and Fixed Random Forest yielded a mean difference of $\mu_d = -0.0341$ ($p = 0.0894$, Cohen's $d = -0.9972$), indicating a noticeable practical performance penalty when using adaptive switching instead of fixed tree ensembles.

**Conclusion**: Operating a fixed tree-based ensemble (Random Forest or XGBoost) provides superior routing accuracy, higher Macro F1, and two orders of magnitude lower decision latency than maintaining an adaptive multi-model switching layer.

---

## 3. Architecture & Candidate Models

```
                          Client HTTP Request
                                  |
                                  v
                    +---------------------------+
                    | Priority/Deadline Policy  |  (Non-invasive outer tier)
                    +---------------------------+
                                  |
                                  v
                    +---------------------------+
                    | Adaptive Model Selector   |
                    |                           |
                    |  Strategy A: Policy       |
                    |  Strategy B: Learned Meta |
                    +---------------------------+
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
          v                       v                       v
     [SVM Router]       [Random Forest Router]    [Fallback Router]
   (Models: LR, RF,      (Selected based on      (Least Connections)
    DT, SVM, XGB)         pre-routing context)
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                                  v
                     Selected Backend Server
                    (Server 1 / Server 2 / Server 3)
```

### 3.1 Five Candidate ML Models
1. **Logistic Regression (`models/logistic_regression.joblib`)**: Scaled $L_2$-regularized linear classifier.
2. **Random Forest (`models/random_forest.joblib`)**: 100-tree ensemble with max depth 5 and min samples split 4.
3. **Decision Tree (`models/decision_tree.joblib`)**: Single interpretable tree with max depth 5.
4. **SVM (`models/svm.joblib`)**: Radial basis function (RBF) kernel with probability calibration.
5. **XGBoost (`models/xgboost.joblib`)**: Gradient-boosted decision trees (100 estimators, max depth 4, learning rate 0.1).

### 3.2 Traditional Routing Baselines
* **Round Robin**: Deterministic sequential cyclic routing.
* **Least Connections**: Routing to minimum active connections with lowest-index tie-breaking.
* **IP Hash**: Deterministic MD5 hashing on client IP.

---

## 4. Adaptive Model Selection Strategies

### 4.1 Strategy A: Evidence-Based Policy Selector
The Evidence-Based Policy Selector derives operational regimes strictly from real-time pre-routing features and assigns the model that performed best on training experiments:
* **Regime Inference Logic**:
  * $\text{CPU Spread} > 40\%$ or $\max(\text{CPU}) > 70\%$ with $\min(\text{CPU}) < 30\% \implies \text{Backend Imbalance} \to \textbf{SVM}$
  * $\text{Total Active Connections} \ge 8 \implies \text{Queue Contention} \to \textbf{Random Forest}$
  * $\text{Mean CPU} > 65\% \implies \text{CPU-Heavy Workload} \to \textbf{Random Forest}$
  * $\text{Total Connections} \ge 5$ or $\text{Mean Response Time} > 50\text{ ms} \implies \text{High Load} \to \textbf{Random Forest}$
  * $\text{Total Connections} \ge 3$ or $\text{Mean Response Time} > 25\text{ ms} \implies \text{Medium Load} \to \textbf{SVM}$
  * Otherwise $\implies \text{Low Load} \to \textbf{SVM}$
* The mapping was fit strictly on training fold observations, avoiding any exposure to held-out test sets.

### 4.2 Strategy B: Learned Meta-Selector
A lightweight, interpretable Decision Tree classifier (`max_depth=4`, `min_samples_split=6`) trained directly on pre-routing context vectors:
$$\mathbf{x}_{\text{context}} \in \mathbb{R}^{15}$$
to predict:
$$\hat{m} \in \{\text{Logistic Regression, Random Forest, Decision Tree, SVM, XGBoost}\}$$
The target label $y_{\text{best\_model}}$ was defined as the candidate model whose prediction matched the ground truth optimal server for that training observation, with deterministic tie-breaking.

---

## 5. Temporal / Experiment-Level Leakage Prevention

To ensure strict zero-leakage evaluation:
1. **GroupKFold by Experiment ID**: Cross-validation was grouped by `experiment_id` across the 37 core experiments. No request from the same experiment was ever present in both training and test folds.
2. **Pre-Routing Only Telemetry**: All context features were collected prior to request forwarding. Forbidden terms (`response_time_ms`, `status_code`, `actual_response_time`, `request_success`, `best_server`) were strictly excluded.
3. **No Test-Set Rule Extraction**: Neither the Policy Selector lookup table nor the Learned Meta-Selector parameters used any information from the test experiments.

---

## 6. Comprehensive 10-Way Comparison Results

### Summary Performance Matrix across 5-Fold GroupKFold Cross-Validation

| Routing Strategy | CV Accuracy Mean $\pm$ Std | CV Macro F1 Mean $\pm$ Std | Weighted F1 | Suboptimal Routing Rate | Model Selection Accuracy | Avg Routing Overhead (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Fixed)** | **0.9344 $\pm$ 0.0651** | **0.9294 $\pm$ 0.0690** | **0.9302** | **0.0656** | 1.0000 | 0.019 ms |
| **XGBoost (Fixed)** | **0.9344 $\pm$ 0.0651** | **0.9294 $\pm$ 0.0690** | **0.9302** | **0.0656** | 1.0000 | 0.012 ms |
| **Decision Tree (Fixed)** | **0.9344 $\pm$ 0.0651** | **0.9294 $\pm$ 0.0690** | **0.9302** | **0.0656** | 1.0000 | **0.004 ms** |
| **SVM (Fixed)** | 0.9265 $\pm$ 0.0642 | 0.8953 $\pm$ 0.0927 | 0.9287 | 0.0735 | 1.0000 | 0.005 ms |
| **Adaptive: Policy Selector** | 0.9265 $\pm$ 0.0642 | 0.8953 $\pm$ 0.0927 | 0.9287 | 0.0735 | **0.9265** | 0.188 ms |
| **Adaptive: Meta Selector** | 0.9265 $\pm$ 0.0642 | 0.8953 $\pm$ 0.0927 | 0.9287 | 0.0735 | **0.9265** | 1.916 ms |
| **Logistic Regression (Fixed)**| 0.8960 $\pm$ 0.0579 | 0.8748 $\pm$ 0.0774 | 0.8906 | 0.1040 | 1.0000 | 0.005 ms |
| *Baseline: IP Hash* | 0.3498 $\pm$ 0.0210 | 0.3322 $\pm$ 0.0183 | 0.3653 | 0.6502 | 0.0000 | 0.006 ms |
| *Baseline: Round Robin* | 0.2974 $\pm$ 0.0368 | 0.2778 $\pm$ 0.0449 | 0.3169 | 0.7026 | 0.0000 | 0.005 ms |
| *Baseline: Least Connections* | 0.0300 $\pm$ 0.0600 | 0.0215 $\pm$ 0.0430 | 0.0161 | 0.9700 | 0.0000 | 0.008 ms |

---

## 7. Model Selection Accuracy vs. Routing Effectiveness

A central question of Phase 13 was:
> *Does correctly identifying the historically best model translate to higher routing effectiveness?*

### Empirical Finding: **NO**
* Both adaptive selectors achieved **92.65% Model Selection Accuracy** (identifying an accurate candidate model in 92.65% of test requests).
* However, because the selector frequently delegated to **SVM** during medium and dynamic regimes, it achieved **89.53% Macro F1**, matching SVM's individual score but failing to achieve the **92.94% Macro F1** delivered by fixed **Random Forest** or **XGBoost**.
* In essence, the meta-selector learned to mimic the training-set winners, but could not outperform the strongest generalizer across unseen test partitions.

---

## 8. Model-Switching Stability & Timeline Analysis

Dynamic stability was evaluated across 4 simulated operational transitions:
1. **Scenario A (Low $\to$ Medium $\to$ High)**: Monotonic transition from SVM (low load) to SVM (medium load) to Random Forest (high load). Switch count: **1**.
2. **Scenario B (Medium $\to$ Burst $\to$ Medium)**: Transient pulse from SVM to Random Forest during the burst, smoothly returning to SVM as queues cleared. Switch count: **2**.
3. **Scenario C (Low $\to$ CPU-Heavy $\to$ High)**: Transitioned from SVM to Random Forest upon compute saturation. Switch count: **1**.
4. **Scenario D (Balanced $\to$ Backend Imbalance)**: Detected asymmetric node degradation ($>55\%$ spread) and assigned SVM. Switch count: **0** (SVM remained active).

Across all 4 scenarios, **zero high-frequency oscillations occurred**. Switching tracked true underlying regime changes smoothly, confirming that hysteresis mechanisms were unnecessary.

---

## 9. Decision Overhead & Latency Impact

| Router Architecture | Selection Overhead | Inference Time | Total Decision Time | Relative Overhead vs Fixed |
| :--- | :---: | :---: | :---: | :---: |
| **Fixed Random Forest** | 0.000 ms | 0.019 ms | **0.019 ms** | **1.0x (Baseline)** |
| **Fixed Decision Tree** | 0.000 ms | 0.004 ms | **0.004 ms** | **0.2x** |
| **Fixed SVM** | 0.000 ms | 0.005 ms | **0.005 ms** | **0.3x** |
| **Adaptive Policy Selector** | 0.170 ms | 0.018 ms | **0.188 ms** | **9.9x slower** |
| **Adaptive Meta-Selector** | 1.897 ms | 0.019 ms | **1.916 ms** | **100.8x slower** |

The Learned Meta-Selector required ~1.9 ms to extract context features, invoke the meta-model, and then invoke the candidate model. Under high-throughput workloads (e.g. $>500\text{ req/s}$), adding 1.9 ms of overhead per routing decision creates CPU serialization at the load balancer.

---

## 10. Confidence & Fallback Handling

The adaptive architecture includes an unconditional safety boundary:
* If the collector is unavailable, backends are unreachable, confidence is $<0.35$, or model inference throws an exception, the system automatically routes via `LeastConnectionsRouter`.
* **Empirical Fallback Rate**: **0.0%** in valid operational regimes.
* **Failure Test**: When all backends were marked unavailable, the system safely diverted traffic to fallback without crashing, setting `X-Adaptive-Fallback: true`.

---

## 11. Priority & Deadline-Aware Extension

Evaluated on the isolated 160-request priority subset:
* **Throughput**: Priority-aware adaptive routing achieved **115.4 req/s** vs. 72.85 req/s for pure adaptive routing (+58.4% gain).
* **Tail Latency (P95)**: Reduced from 105.0 ms to **88.0 ms (-16.2% tail reduction)**.
* **Deadline Violations**: 20 requests out of 80 (25.0% violation rate), consistent with Phase 10 baselines.

---

## 12. Matched Paired Statistical Hypothesis Testing

Matched paired t-tests across the 5 cross-validation folds:

| Paired Comparison | Sample Size ($N$) | Mean Difference ($\mu_d$) | 95% Confidence Interval | $t$-Statistic | $p$-Value | Statistically Significant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adaptive Policy vs SVM** | 5 folds | $0.0000$ | $[0.0000, 0.0000]$ | N/A | N/A | Equivalent |
| **Adaptive Policy vs Random Forest** | 5 folds | $-0.0341$ | $[-0.0766, +0.0084]$ | $-2.2280$ | $0.0894$ | No ($p > 0.05$) |
| **Adaptive Policy vs Logistic Regression** | 5 folds | $+0.0205$ | $[-0.0421, +0.0831]$ | $+0.9200$ | $0.4101$ | No ($p > 0.05$) |
| **Adaptive Meta vs SVM** | 5 folds | $0.0000$ | $[0.0000, 0.0000]$ | N/A | N/A | Equivalent |
| **Adaptive Policy vs Adaptive Meta** | 5 folds | $0.0000$ | $[0.0000, 0.0000]$ | N/A | N/A | Equivalent |

While differences between Adaptive Policy and Random Forest did not cross the strict threshold of $p < 0.05$ due to fold variance, the effect size is substantial (Cohen's $d = -0.9972$, a large negative effect indicating that Random Forest consistently outperformed adaptive policy across folds).

---

## 13. Artifact & Visualization Index

All Phase 13 artifacts are persisted:

### Data & Results
* Meta-Dataset: [`data/phase13/meta_dataset.csv`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase13/meta_dataset.csv)
* Phase 13 Raw Dataset: [`data/phase13/phase13_raw.csv`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase13/phase13_raw.csv)
* Summary Performance Matrix: [`data/phase13/phase13_summary.csv`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase13/phase13_summary.csv)
* Experiment Manifest: [`data/phase13/experiment_manifest.json`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase13/experiment_manifest.json)
* Environment Specifications: [`data/phase13/environment.json`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase13/environment.json)
* Statistical Analysis JSON: [`data/phase13/statistical_analysis.json`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase13/statistical_analysis.json)

### Research Visualizations (12 Publication Plots)
1. [`experiments/results/phase13/01_fixed_vs_adaptive_performance.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/01_fixed_vs_adaptive_performance.png) - Accuracy and Macro F1 comparison across 10 strategies
2. [`experiments/results/phase13/02_model_selection_frequency.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/02_model_selection_frequency.png) - Model selection share distribution
3. [`experiments/results/phase13/03_model_selection_by_scenario.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/03_model_selection_by_scenario.png) - Model assignment per operational regime
4. [`experiments/results/phase13/04_adaptive_vs_fixed_macro_f1.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/04_adaptive_vs_fixed_macro_f1.png) - Macro F1 ranking
5. [`experiments/results/phase13/05_adaptive_vs_fixed_accuracy.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/05_adaptive_vs_fixed_accuracy.png) - Cross-validation Accuracy ranking
6. [`experiments/results/phase13/06_latency_comparison.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/06_latency_comparison.png) - End-to-end response time distributions
7. [`experiments/results/phase13/07_throughput_comparison.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/07_throughput_comparison.png) - Live throughput comparison
8. [`experiments/results/phase13/08_model_switching_timeline.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/08_model_switching_timeline.png) - Dynamic transition timeline
9. [`experiments/results/phase13/09_selector_confusion_matrix.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/09_selector_confusion_matrix.png) - Meta-selector model assignment matrix
10. [`experiments/results/phase13/10_adaptive_overhead.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/10_adaptive_overhead.png) - Decision latency overhead
11. [`experiments/results/phase13/11_fallback_analysis.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/11_fallback_analysis.png) - Fallback rate analysis
12. [`experiments/results/phase13/12_priority_adaptive_comparison.png`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/experiments/results/phase13/12_priority_adaptive_comparison.png) - Priority vs pure adaptive routing

---

## 14. Final Research Conclusion

> **Does context-aware model selection provide a measurable advantage over using a single fixed ML model?**

### **NO.**

The empirical investigation definitively concludes that **adaptive model selection does NOT provide a measurable advantage over using a single fixed Random Forest or XGBoost model**:
1. Fixed tree ensembles achieve **higher accuracy (93.44% vs 92.65%)** and **higher Macro F1 (92.94% vs 89.53%)** than adaptive selection layers.
2. Adaptive model selection introduces **10x to 100x higher routing decision overhead** (0.188–1.916 ms vs. 0.004–0.019 ms for fixed models).
3. The added architectural complexity of training and maintaining meta-models does not yield superior cluster throughput or lower tail latency.
4. **Recommendation**: Subsequent project phases should deploy a **fixed Random Forest or XGBoost model** rather than an adaptive multi-model selector.
