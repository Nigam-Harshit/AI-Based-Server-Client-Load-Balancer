# Phase 9 Research Report: Live ML vs Traditional Load-Balancing Experiment

## 1. Objective

The primary objective of Phase 9 was to conduct a rigorous, controlled, and reproducible empirical comparison between traditional load-balancing algorithms (**Round Robin**, **Least Connections**, **IP Hash**) and the **Machine Learning (Logistic Regression)** driven load balancer integrated in Phase 8.

The study investigates whether the ML-assisted routing model provides measurable system-level advantages (throughput, median/tail latency, CPU utilization, resilience under stress) over traditional deterministic baselines under identical live server, client, and network conditions.

> [!IMPORTANT]
> **Research Boundary Adherence**:
> - The production Logistic Regression model artifact (`models/logistic_regression.joblib`) was strictly preserved and evaluated without retraining.
> - The 15 active pre-routing features and Phase-5 dataset were strictly untouched.
> - No artificial priority/deadline or SLA scheduling logic was added.
> - All observations, including failure cases and model routing limitations, are recorded faithfully without selective exclusion.

---

## 2. Experimental Architecture

The benchmark was executed against the project's distributed server-client testbed:
```text
                          ┌── Backend Server 1 (:8001)
                          │
Client Workload ──► Load Balancer (:8000) ──┼── Backend Server 2 (:8002)
                          │
                          └── Backend Server 3 (:8003)
```

- **Backend Cluster**: 3 identical HTTP server instances (`server/app.py`) running standard-library `ThreadingHTTPServer` with simulated CPU processing endpoints (`GET /process?duration=...`) and non-blocking runtime metric probes (`GET /metrics`).
- **Load Balancer**: Centralized proxy (`load_balancer/app.py`) dynamically configured per run with the target algorithm (`round_robin`, `least_connections`, `ip_hash`, `ml`).
- **Client Workload Generator**: Multi-threaded workload dispatcher (`client/workload_generator.py`) driving concurrent requests across pre-configured scenario templates.

---

## 3. Hardware & Software Environment

As automatically recorded in `data/phase9/environment.json`:
- **Operating System**: Windows 11 (build 10.0.26200-SP0)
- **Python Runtime**: 3.12.4 (MSC v.1940 64-bit AMD64)
- **Processor**: Intel/AMD x86_64, 16 Logical Cores (12 Physical Cores)
- **Host Memory**: 15.63 GB Physical RAM
- **Backend Endpoints**: `http://127.0.0.1:8001`, `http://127.0.0.1:8002`, `http://127.0.0.1:8003`
- **Load Balancer Endpoint**: `http://127.0.0.1:8000`
- **Trained Model Artifact**: `models/logistic_regression.joblib` (`StandardScaler` + `LogisticRegression`)

---

## 4. Workload Scenarios & Parameters

All 7 required scenarios were tested with deterministic random seeds across 3 independent repetitions ($N = 3 \times 4 \times 7 = 84$ total benchmark runs):

| Scenario | Requests | Concurrency | Dispatch Rate | Request Duration | Pattern / Characteristics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **A: Low Traffic** | 20 | 2 | 5.0 req/s | 0.02s | Mild, paced background traffic |
| **B: Medium Traffic** | 50 | 5 | 15.0 req/s | 0.03s | Moderate sustained concurrent flow |
| **C: High Traffic** | 100 | 15 | Unthrottled | 0.03s | Heavy saturation, concurrent load spikes |
| **D: Burst Traffic** | 60 | 20 | Unthrottled | 0.04s | Bursts of 20 reqs with 0.3s inter-burst delay |
| **E: CPU Heavy** | 30 | 5 | Unthrottled | 0.12s | Long-running computational workloads |
| **F: Mixed** | 60 | 6 | 20.0 req/s | 0.04s | Stochastically mixed `/health` and `/process` |
| **G: Dynamic** | 80 | 10 | Unthrottled | 0.03s | High concurrency with fluctuating arrival rates |

---

## 5. Dataset Description

The Phase-9 experimental dataset was written directly to `data/phase9/` and kept completely separated from Phase 5:
- [`data/phase9/raw_results.csv`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase9/raw_results.csv): 4,800 individual request execution records. Each record logs `request_id`, `algorithm`, `scenario`, `response_time_ms`, `success`, `status_code`, `backend_server`, `ml_predicted_server`, `ml_confidence`, `ml_fallback`, all 15 pre-routing metric fields, and the theoretical `best_server` label.
- [`data/phase9/run_summaries.csv`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase9/run_summaries.csv): 84 aggregate run summaries reporting throughput, latency distributions (P50, P95, P99, Max), success rates, CPU utilization during workloads, and routing balance.
- [`data/phase9/edge_conditions.json`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase9/edge_conditions.json): Measurements from server shutdown failure, failover routing, recovery, and sharp 25-worker burst stress tests.
- [`data/phase9/statistical_analysis.json`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase9/statistical_analysis.json): Descriptive and significance statistics across all configurations.

---

## 6. Experimental Results & Performance Comparison

### 6.1 Scenario-Wise Performance Table
Means across 3 independent repetitions for each algorithm and scenario:

| Scenario | Algorithm | Throughput (RPS) | P50 Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | Success Rate (%) | CPU Util (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Low Traffic** | Round Robin | 4.99 | 51.32 | 68.32 | 78.61 | 100.0% | 1.44% |
| | Least Connections | 4.98 | 52.16 | 62.64 | 67.94 | 100.0% | 1.16% |
| | IP Hash | 4.99 | 50.19 | 65.44 | 66.14 | 100.0% | 1.42% |
| | **ML (Logistic Reg)** | 4.99 | 79.57 | 111.13 | 124.43 | 100.0% | 1.88% |
| **Medium Traffic**| Round Robin | **14.89** | **48.13** | **69.16** | 78.09 | 100.0% | 1.83% |
| | Least Connections | 14.86 | 53.96 | 73.11 | 82.18 | 100.0% | 1.64% |
| | IP Hash | 14.88 | 54.81 | 73.63 | 77.08 | 100.0% | 2.00% |
| | **ML (Logistic Reg)** | 14.85 | 79.51 | 116.34 | 127.30 | 100.0% | 3.06% |
| **High Traffic** | Round Robin | **206.23** | 44.28 | 59.63 | 394.98 | 100.0% | 8.38% |
| | Least Connections | 172.05 | **41.15** | 58.25 | **227.30** | 100.0% | 7.33% |
| | IP Hash | 201.10 | 45.06 | **53.90** | 230.93 | 100.0% | 9.69% |
| | **ML (Logistic Reg)** | 49.57 | 290.20 | 369.03 | 388.95 | 100.0% | 3.68% |
| **Burst Traffic**| Round Robin | 49.20 | 65.52 | 573.20 | 578.89 | 100.0% | 3.71% |
| | Least Connections | 44.08 | 66.19 | 573.42 | 654.32 | 100.0% | 3.13% |
| | IP Hash | **49.92** | **65.10** | **570.39** | **577.77** | 100.0% | 3.66% |
| | **ML (Logistic Reg)** | 44.40 | 228.20 | 777.15 | 816.64 | 100.0% | 4.85% |
| **CPU Heavy** | Round Robin | 34.89 | 139.15 | 152.60 | 159.28 | 100.0% | 3.31% |
| | Least Connections | 35.15 | 136.79 | 151.94 | **155.91** | 100.0% | 2.95% |
| | IP Hash | **35.91** | **134.68** | **151.61** | 159.04 | 100.0% | 4.09% |
| | **ML (Logistic Reg)** | 26.96 | 169.81 | 228.09 | 242.52 | 100.0% | 3.85% |
| **Mixed** | Round Robin | 19.51 | **36.50** | **112.55** | **120.76** | 100.0% | 2.05% |
| | Least Connections | 19.49 | 44.34 | 115.13 | 121.24 | 100.0% | 2.05% |
| | IP Hash | **19.56** | 41.31 | 114.71 | 121.41 | 100.0% | 2.22% |
| | **ML (Logistic Reg)** | 19.34 | 66.02 | 138.92 | 160.33 | 100.0% | 4.21% |
| **Dynamic** | Round Robin | **214.14** | **41.29** | 57.68 | **62.64** | 100.0% | 11.31% |
| | Least Connections | 208.66 | 41.71 | **57.24** | 62.75 | 100.0% | 10.95% |
| | IP Hash | 211.68 | 41.96 | 58.21 | 61.00 | 100.0% | 9.34% |
| | **ML (Logistic Reg)** | 50.00 | 190.64 | 251.86 | 277.09 | 100.0% | 4.74% |

---

### 6.2 Overall Metric Summary Across All 84 Runs

- **Best Overall Throughput**: **Round Robin** ($77.69\text{ RPS}$) closely followed by **IP Hash** ($76.86\text{ RPS}$) and **Least Connections** ($71.32\text{ RPS}$). ML achieved $30.02\text{ RPS}$.
- **Best Median Latency (P50)**: **Round Robin** ($60.88\text{ ms}$) and **IP Hash** ($61.87\text{ ms}$). ML recorded $157.71\text{ ms}$.
- **Best Tail Latency (P95)**: **IP Hash** ($155.41\text{ ms}$) and **Least Connections** ($155.96\text{ ms}$). ML recorded $284.65\text{ ms}$.
- **Best Worst-Case Latency (P99)**: **IP Hash** ($184.77\text{ ms}$) and **Least Connections** ($195.95\text{ ms}$). ML recorded $305.32\text{ ms}$.
- **Best Success Rate**: **All algorithms achieved 100.0% success** (0 failed requests across 4,800 executed requests).
- **CPU Efficiency**: ML exhibited lowest average backend CPU consumption ($3.75\%$) compared to Least Connections ($4.17\%$) and Round Robin ($4.58\%$) primarily due to lower delivered throughput under saturated concurrency scenarios.

---

## 7. Machine Learning Routing Analysis

### 7.1 Prediction Distribution & Class Imbalance Collapse
- **Total ML Routing Invocations**: 1,200 requests.
- **Predicted Backend**:
  - `http://127.0.0.1:8003`: **1,200 requests (100.0%)**
  - `http://127.0.0.1:8001`: **0 requests (0.0%)**
  - `http://127.0.0.1:8002`: **0 requests (0.0%)**
- **Average Model Confidence**: **1.0000** (saturated maximum confidence across all requests).
- **Fallback Triggered**: **0 runs (0.0%)** under normal operational benchmark.

### 7.2 Root Cause Analysis: Why ML Pinned Server 3
In-depth inspection of the production Logistic Regression weights and training distribution revealed the root cause:
1. In the Phase-5 dataset, Server 3 was underutilized in several recording sessions, leading the automated label generator to assign `best_server = server-3` disproportionately during burst and high-concurrency periods.
2. The fitted Logistic Regression model learned large positive weights for `server_1_connections` ($+1.58$) and negative weights for `server_3_connections` ($-1.15$), `server_3_response_time` ($-1.17$), and `server_3_network_latency` ($-1.31$).
3. Under live traffic, as soon as `server_1` and `server_2` exhibited slight loopback probe latencies, the model's uncalibrated linear decision boundary saturated $P(\text{class}=2) \approx 1.0$.
4. **Self-Reinforcing Serialization**: By routing every single incoming request to Server 3, the ML load balancer effectively turned a 3-node cluster into a **single-node bottleneck**. While Server 1 and Server 2 sat completely idle, Server 3 queued all requests sequentially, multiplying P50 latency from $\sim 44\text{ms}$ up to $290\text{ms}$ under High Traffic, and capping throughput at $49.57\text{ RPS}$ vs Round Robin's $206.23\text{ RPS}$.

---

## 8. High-Load & Stress Analysis

Under low traffic (Scenario A, 5 req/s), all algorithms performed acceptably ($4.99\text{ RPS}$), although ML had higher latency ($79.57\text{ ms}$ vs Round Robin's $51.32\text{ ms}$) due to the synchronous HTTP metric probing overhead before each decision.

However, under escalating load:
- **Throughput Collapse**: In High Traffic (15 concurrent workers) and Dynamic Traffic (10 concurrent workers), Round Robin and Least Connections effectively scaled horizontally across all 3 backend nodes, delivering $>200\text{ RPS}$. In contrast, ML peaked at $\sim 50\text{ RPS}$—a **4x throughput deficit**.
- **Latency Blowup**: In High Traffic, ML median latency escalated by **+555%** ($290.20\text{ ms}$ vs Round Robin's $44.28\text{ ms}$).
- **Non-Linear Degradation**: In Burst Traffic, worst-case tail latency on ML jumped to $816.64\text{ ms}$ vs $578.89\text{ ms}$ on Round Robin.

---

## 9. Edge Conditions & Fault Tolerance

The edge experiments documented in [`data/phase9/edge_conditions.json`](file:///c:/AIML/AI-Based%20Server-Client%20Load%20Balancer/data/phase9/edge_conditions.json) verified the system's fault-tolerance guarantees:

### 9.1 Backend Failure & Automatic Failover
1. **Server 3 Offline**: To test recovery from ML bias, Server 3 (the server ML predicted 100% of the time) was forcibly terminated.
2. **Detection & Routing**: The live metrics probe immediately flagged `http://127.0.0.1:8003` as `available: False`.
3. **Adaptive ML Decision**: The feature vector for Server 3 was zeroed, shifting the model's prediction dynamically to Server 1 (`http://127.0.0.1:8001`).
4. **Zero Downtime**: 10 out of 10 requests succeeded ($100.0\%$ success rate, status code 200). The load balancer did not crash, throw unhandled exceptions, or forward any requests to the dead server.

### 9.2 Service Recovery
When Server 3 was restarted, subsequent health probes immediately verified connectivity and returned the node to active participation without requiring a load balancer restart.

### 9.3 Short High-Concurrency Burst (25 Workers)
Under an instantaneous 25-concurrency burst of 50 requests:
- **Round Robin**: $37.50\text{ RPS}$, P50 latency $61.45\text{ ms}$, P95 latency $578.87\text{ ms}$, 100% success.
- **Least Connections**: $37.45\text{ RPS}$, P50 latency $64.54\text{ ms}$, P95 latency $571.79\text{ ms}$, 100% success.
- **ML Router**: $44.99\text{ RPS}$, P50 latency $391.53\text{ ms}$, P95 latency $906.06\text{ ms}$, 100% success.
*ML achieved completion of the burst but suffered severe latency degradation (P50 was 6.3x higher).*

---

## 10. Research Visualizations

The following 10 research-quality figures were produced and archived in `docs/plots/`:

1. [`01_throughput_by_scenario.png`](docs/plots/01_throughput_by_scenario.png): Throughput (RPS) comparison across all 7 scenarios.
2. [`02_p50_latency.png`](docs/plots/02_p50_latency.png): Median response time (ms) illustrating ML's latency penalty under load.
3. [`03_p95_latency.png`](docs/plots/03_p95_latency.png): Tail latency (ms) highlighting queue delays on the bottlenecked server.
4. [`04_p99_latency.png`](docs/plots/04_p99_latency.png): Worst-case P99 latency comparisons.
5. [`05_cpu_utilization.png`](docs/plots/05_cpu_utilization.png): Backend CPU consumption during workloads.
6. [`06_backend_distribution.png`](docs/plots/06_backend_distribution.png): Total request volume per backend server (illustrating Round Robin's uniform 33%/33%/33% distribution vs ML's 100% concentration on Server 3).
7. [`07_ml_confidence_distribution.png`](docs/plots/07_ml_confidence_distribution.png): Probability histogram demonstrating extreme confidence saturation ($1.0$).
8. [`08_ml_fallback_rate.png`](docs/plots/08_ml_fallback_rate.png): Fallback rate tracking across scenarios ($0\%$ during healthy benchmarks).
9. [`09_performance_vs_workload_intensity.png`](docs/plots/09_performance_vs_workload_intensity.png): Scaling curve showing divergent latency growth as concurrency escalates.
10. [`10_algorithm_robustness.png`](docs/plots/10_algorithm_robustness.png): Coefficient of Variation (%) across independent repetitions.

---

## 11. Statistical Significance

Welch's two-sample unequal-variance t-tests evaluated the difference between ML and the Round Robin baseline across repetitions:
- **P50 Latency**: In High Traffic, $t = 78.43$, $p < 0.0001$ (statistically significant at $\alpha = 0.01$). ML had a mean P50 latency $+245.92\text{ ms}$ higher than Round Robin.
- **Throughput**: In High Traffic, $t = -33.91$, $p < 0.0001$ (statistically significant at $\alpha = 0.01$). ML throughput was $156.66\text{ RPS}$ lower than Round Robin.
- **Low Traffic**: Mean difference was $+28.25\text{ ms}$ ($p = 0.027$, statistically significant at $\alpha = 0.05$), reflecting the probe overhead.

> [!NOTE]
> While $N=3$ repetitions per scenario provides strong directional evidence of performance degradation due to single-server pinning, broader asymptotic claims across variable network topologies warrant larger sample sizes.

---

## 12. Major Research Findings

1. **ML Did NOT Outperform Traditional Baselines in Live Execution**:
   Although Logistic Regression achieved 75.6% cross-validation accuracy on the static Phase-5 tabular dataset, in a live closed-loop distributed system it severely degraded performance compared to Round Robin and Least Connections.
2. **The Feedback Loop Collapse (Static Classifier Fallacy)**:
   A static classifier trained on independent observations assumes that choosing a backend does not alter future observations. In reality, routing is a **stateful dynamic control problem**. By consistently picking what it thought was the "optimal" server, the classifier overloaded that exact server, turning it into the worst performer while leaving the other servers idle.
3. **Synchronous Metric Probing Overhead**:
   Querying `/metrics` across 3 servers synchronously inside the HTTP routing path adds a non-trivial baseline latency floor ($\sim 15\text{--}25\text{ms}$) per request compared to $O(1)$ in-memory counter operations in Round Robin or Least Connections.
4. **Fault Tolerance and Safety Verified**:
   The fallback mechanism proved robust: when Server 3 was killed, the ML router immediately adapted, avoided the offline node, and maintained 100% request availability.

---

## 13. Limitations

1. **Point-in-Time Metric Polling**: Metric probes were executed synchronously per request. In production high-throughput load balancers, metrics are collected asynchronously via background heartbeats or eBPF.
2. **Fixed Synthetic Dataset Bias**: The ML model was trained on Phase 5 data without reinforcement learning or online adaptive feedback loops.
3. **Homogeneous Backends**: All backends ran on local loopback on the same physical host; real-world heterogeneous server hardware across WAN links may expose different tradeoffs.
4. **Priority/Urgency Absence**: All requests carried equal weight without deadline or priority classes.

---

## 14. Future Experiment: Priority & SLA-Aware ML Routing

> [!IMPORTANT]
> **Priority/deadline-aware routing was NOT included in this experiment.**

A natural subsequent research extension is to formulate load balancing not merely as a server-load classifier, but as a **multi-objective priority scheduler**:
- Inputs: Request Priority Class (e.g. `High`, `Normal`, `Batch`), Deadline Remaining ($\Delta t$), Request Payload Size, alongside Server State.
- Objective: Evaluate whether ML can outperform traditional priority queues (e.g., Weighted Fair Queueing or Priority-Least-Connections) when backends are saturated and low-priority requests must be deferred to satisfy high-priority SLAs.

