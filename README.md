# AI-Based Server-Client Load Balancer

## Overview
An experimental study evaluating whether machine-learning-assisted server selection can improve load balancing performance compared to conventional load balancing strategies in a distributed environment.

## High-Level Methodology
1. **Infrastructure Baseline**: Distributed setup consisting of a client workload generator, a load balancer, and three identical HTTP backend servers.
2. **Monitoring System**: Metrics collection capturing server load, latency, CPU utilization, and request throughput.
3. **Experimental Dataset**: Real measurements generated from controlled baseline workloads.
4. **Conventional Baseline**: Round Robin server selection evaluated under synthetic and realistic request patterns.
5. **Machine Learning Model**: Random Forest model trained on tabular metric data for predictive server selection.
6. **ML-Assisted Load Balancing**: Intelligent request routing informed by Random Forest predictions.
7. **Performance Comparison**: Empirical analysis comparing conventional vs. ML-assisted load balancing.

---

## Architecture Overview
```text
          [Controlled Workload Generator]
                         │
                         │ HTTP Requests
                         ▼
                 ┌───────────────┐
                 │ Load Balancer │ ── [MetricsCollector]
                 │     :8000     │           │
                 └───────┬───────┘           │ (Polls /metrics)
                         │                   ▼
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
          Server 1   Server 2   Server 3
           :8001      :8002      :8003
```

---

## Phase 1 & 2 Summary

1. **Backend Server Nodes (`server/app.py`)**:
   - Reusable HTTP server instances implemented with Python standard library `ThreadingHTTPServer`.
   - Endpoints:
     - `GET /health`: Liveness probe returning server ID, status, and port.
     - `GET /process?duration=...`: Simulated computational workload returning completion status and duration.
     - `GET /metrics`: Real-time node metrics.
   - Configurable via CLI arguments (`--id`, `--host`, `--port`, `--workers`).

2. **Backend Configuration (`config/backends.py`)**:
   - Centralized list of backend URLs (`DEFAULT_BACKENDS = ["http://127.0.0.1:8001", "http://127.0.0.1:8002", "http://127.0.0.1:8003"]`).

3. **Routing Abstraction (`load_balancer/router.py`)**:
   - `BaseRouter`: Abstract base class providing `select(client_ip)` and `release(backend, success)` lifecycle hooks, along with a `route(client_ip)` context manager.
   - **Round Robin (`RoundRobinRouter`)**: Thread-safe sequential cyclic routing.
   - **Least Connections (`LeastConnectionsRouter`)**: Routes to lowest active connection count with deterministic lowest-index tie-breaking.
   - **IP Hash (`IPHashRouter`)**: Deterministic client IP routing via `hashlib.md5`.

4. **Central Load Balancer Server (`load_balancer/app.py`)**:
   - Listens on `:8000`, routes incoming client requests, preserves client IP via `X-Forwarded-For`, and adds `X-Backend-Server` response header.
   - Handles backend failures returning `502 Bad Gateway` and timeouts returning `504 Gateway Timeout`.

---

## Phase 3: Real-Time Metric Collection

### Metric Definitions and Measurement Methodology

| Metric | Unit | Measurement Mechanism | Sampling / Averaging Behavior |
| :--- | :--- | :--- | :--- |
| **CPU Utilization** | `%` (`0.0` – `100.0`) | `psutil.Process().cpu_percent(interval=None)` | Measures process-level CPU consumption between successive calls without blocking. |
| **Memory Utilization** | `%` (`0.0` – `100.0`) | `psutil.Process().memory_percent()` | Physical RAM percentage occupied by the server process relative to total system memory. |
| **Active Connections** | Count (`int >= 0`) | Server-side concurrency counter protected by `threading.Lock` | Incremented when request acquires execution worker slot; decremented in `finally` upon request completion. |
| **Recent Avg Response Time** | `ms` (`float >= 0.0`) | High-resolution wall-clock timer `time.perf_counter()` per request | Moving average over the last 50 completed requests tracked via a bounded `collections.deque`. |
| **Network Latency** | `ms` (`float >= 0.0`) | Probe round-trip time (`time.perf_counter()`) measured by collector; local socket TCP handshake on server | Collector measures real network RTT during probe requests; server measures local loopback connect latency with 1s caching. |
| **Request Queue Length** | Count (`int >= 0`) | Server concurrency queue counter protected by `threading.Lock` | Incremented when request enters server waiting for an available worker thread semaphore slot; decremented once slot is acquired. |

---

## Phase 4: Controlled Workload Generator

### Objective
Create a reproducible traffic-generation engine capable of generating controlled traffic against local backend servers or the load balancer, verifying that varying traffic patterns produce measurable changes in the Phase-3 runtime metrics.

### Workload Generator Architecture (`client/workload_generator.py`)
```text
WorkloadConfig (parameters, scenario template, random seed)
                    │
                    ▼
          WorkloadGenerator
                    │
   ThreadPoolExecutor (concurrency workers)
        │           │           │
        ▼           ▼           ▼
    HTTP Req    HTTP Req    HTTP Req
        │           │           │
        └───────────┬───────────┘
                    ▼
     Target (Load Balancer / Server)
                    │
                    ▼
        RequestRecords & Summary
```

### Configurable Parameters
- `target_url`: Base URL of target (Load Balancer `:8000` or Server `:8001`).
- `num_requests`: Total request count.
- `concurrency`: Worker thread pool capacity.
- `request_rate`: Optional target requests/sec throttle (or unthrottled).
- `endpoint`: `/process`, `/health`, or `mixed`.
- `request_duration`: Simulated execution duration for `/process?duration=...`.
- `burst_size`: Requests per batch for burst traffic.
- `burst_interval`: Interval between bursts in seconds.
- `seed`: Fixed random seed ensuring deterministic request distributions.

### Supported Scenarios

| Scenario | Requests | Concurrency | Rate (req/s) | Endpoint | Duration (s) | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `low_traffic` | 20 | 2 | 5.0 | `/process` | 0.02 | Baseline low-volume continuous traffic. |
| `medium_traffic` | 50 | 5 | 15.0 | `/process` | 0.03 | Moderate sustained concurrent traffic. |
| `high_traffic` | 100 | 15 | None | `/process` | 0.03 | High-concurrency unthrottled traffic. |
| `burst_traffic` | 60 | 20 | None | `/process` | 0.04 | Batches of 20 concurrent requests with pauses. |
| `cpu_heavy` | 30 | 5 | None | `/process` | 0.12 | Longer CPU processing workloads. |
| `mixed` | 60 | 6 | 20.0 | `mixed` | Variable | Pseudo-random mix of `/health` and `/process`. |
| `dynamic` | 80 | 10 | None | `/process` | 0.03 | Scaled multi-worker traffic profile. |

All scenarios support parameter overrides programmatically or via CLI.

### Reproducibility & Safety Constraints
- **Reproducibility**: Mixed request choices and timings rely on seeded pseudo-random number generators (`random.Random(seed)`).
- **Safety**: Traffic targets exclusively localhost (`127.0.0.1`). Workloads are bounded in requests, concurrency, and duration; all resources are released cleanly upon completion.
- **Log Isolation**: Individual `RequestRecord` objects (latencies, status codes, timestamps) represent generator-side client logs and are strictly decoupled from the future Phase-5 experimental dataset.

### Observed Metric Behavior
During experimental validation against 3 local servers behind the load balancer under `burst_traffic`:
- **Before Workload**: Active connections: `0`, Average response time: `0.0ms`, CPU utilization: `~3-4%`.
- **During Workload**: Active connections rose to `4` (server-1), `4` (server-2), `3` (server-3); CPU consumption spiked to `28-43%` across nodes.
- **After Workload**: Active connections returned to `0`; average response times settled at `~80.9ms` matching the workload duration.

---

## Running Tests
Run all unit and integration tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Or with pytest:
```bash
python -m pytest
```

### Running Workload Generator via CLI
```bash
python -m client.workload_generator --scenario low_traffic --target http://127.0.0.1:8000
python -m client.workload_generator --scenario burst_traffic --requests 30 --concurrency 10
```

---

## Phase 5: Experimental Data Collection

### Objective
Establish a reproducible pipeline that executes controlled workload scenarios across traditional routing algorithms (`round_robin`, `least_connections`, `ip_hash`), captures real-time pre-routing cluster state, records post-routing outcomes, assigns data-driven target labels, and validates the resulting dataset.

### Pipeline Architecture
```text
Experiment Configuration (Scenario, Algorithm, Repetitions, Seed)
                            │
                            ▼
                Controlled Workload Generator
                            │
                            ▼
                      Load Balancer
                     /      |      \
                    ▼       ▼       ▼
                Server 1  Server 2  Server 3
                    \       |       /
                     ▼      ▼      ▼
                Real-Time Node Metrics
                            │
                            ▼
                   Experiment Recorder
                  /                   \
                 ▼                     ▼
          data/raw/             data/processed/
      (Raw Observations)     (Validated & Labeled)
                 │                     │
                 └──────────┬──────────┘
                            ▼
                      Data Validator
                            │
                            ▼
                   Data Quality Report
```

### Critical Temporal Separation (No Data Leakage)
- For every incoming request, the complete cluster state (18 metrics: CPU, memory, active connections, response time, network latency, and queue length across all 3 servers) is captured **strictly before the routing decision**.
- Routing outcome metrics (`actual_response_time`, `request_success`) and target labels (`best_server`) are populated only after request completion.
- Feature vectors strictly avoid future leakage.

### Label Generation Strategy
- Labels are assigned independently of the algorithm's choice (`selected_server`).
- Expected service cost is evaluated:
  $$\text{Cost}(S_i) = \text{network\_latency}_i + \text{response\_time}_i \times (1 + \text{connections}_i + \text{queue\_length}_i)$$
- The candidate server with minimum cost is labeled `best_server`.
- When all servers exhibit near-identical cost (difference $\le 2\text{ms}$ on an idle cluster), the label is explicitly set to `None` rather than arbitrarily inventing a preference.

### Data Quality Verification
- Automated validation via `experiments.validator.DataValidator` verifies:
  - 100% column presence across all 30 fields.
  - Monotonic timestamp ordering (`timestamp <= request_start <= request_end`).
  - Strict range boundaries for percentages $[0, 100]$, counts $\ge 0$, and durations $\ge 0$.
  - Zero duplicate records per experiment.
  - Absence of missing values on mandatory feature fields.

### Executing Data Collection via CLI
```bash
python -m experiments.runner --scenario low_traffic --algorithm round_robin --runs 2
```

---

## Phase 8: ML-Driven Load Balancer Integration

### Objective
Integrate the trained ML classification model into the central HTTP Load Balancer with dynamic metric querying, safety fallbacks, pluggable model architectures, and full observability headers.

### Key Capabilities
1. **Integrated Machine Learning Router (`load_balancer/router.py`)**:
   - `MLRouter` queries real-time metrics across backend servers and builds the exact 15-feature input representation.
   - Evaluates probability distributions using `predict_proba` to calculate decision confidence.
   - Preserves modularity: Supports `LogisticRegression` (primary baseline from Phase 7/7.5), `RandomForestClassifier`, or any joblib model artifact.
2. **Safety & Fault Tolerance**:
   - Live availability checking prevents routing to down or degraded nodes.
   - Deterministic fallback to `LeastConnectionsRouter` if predictions target an offline node or if inference encounters an exception.
3. **Observability**:
   - Automatically attaches `X-ML-Predicted-Server`, `X-ML-Confidence`, and `X-ML-Fallback` HTTP response headers.

### Starting the Load Balancer with ML Routing
```bash
# Start Load Balancer using ML routing (default Logistic Regression model)
python -m load_balancer.app --port 8000 --algorithm ml

# Specify an alternative model artifact
python -m load_balancer.app --port 8000 --algorithm ml --model-path models/random_forest.joblib
```
Detailed integration documentation is available in [docs/ml_integration.md](docs/ml_integration.md).

---

## Phase 9: Live ML vs Traditional Load-Balancing Experiment

### Objective
Conduct an empirical, reproducible live performance comparison between **Round Robin**, **Least Connections**, **IP Hash**, and **ML (Logistic Regression)** across 7 standardized workload scenarios with repeated runs and live cluster measurements.

### Key Empirical Findings
- **Throughput Winner**: **Round Robin** achieved top throughput ($77.69\text{ RPS}$ mean, peaking at $214.14\text{ RPS}$ in Dynamic traffic), closely followed by **IP Hash** ($76.86\text{ RPS}$) and **Least Connections** ($71.32\text{ RPS}$).
- **Latency Winner**: **Round Robin** achieved lowest overall median latency ($60.88\text{ ms}$ P50), while **IP Hash** ($155.41\text{ ms}$) and **Least Connections** ($155.96\text{ ms}$) delivered superior tail stability (P95).
- **ML Limitation in Closed-Loop Serving**: The static Logistic Regression classifier suffered from feedback loop collapse: predicting Server 3 consecutively overloaded that single backend, causing queue serialization and throughput degradation ($30.02\text{ RPS}$, P50 $157.71\text{ ms}$).
- **Fault-Tolerance & Edge Conditions**: Zero requests were lost (100% success rate across all 4,800 requests). When a target server was taken offline, the load balancer's live availability probe seamlessly diverted traffic with 0 dropped requests.

### Running the Phase 9 Benchmark Suite
```bash
# Run full live comparison benchmark across all 4 algorithms and 7 scenarios
python -m benchmarks.live_comparison --repetitions 3

# Run dedicated failure and burst stress edge tests
python -m benchmarks.edge_tests

# Generate statistical reports and research plots
python -m benchmarks.analysis
```
Detailed findings and plots are available in [docs/live_performance_comparison.md](docs/live_performance_comparison.md).

---

## Phase 10: Priority & Deadline-Aware Routing

### Objective
Extend the load-balancing system with request-level **priority tiers** (`LOW`, `NORMAL`, `HIGH`, `CRITICAL`) and **deadline slack awareness** ($S = T_{\text{deadline}} - T_{\text{now}} - \hat{D}_{\text{proc}}$) via a non-invasive arbitration layer (`PriorityDeadlineRouter`) wrapping the underlying ML model.

### Key Architectural Capabilities
1. **Multi-Factor Arbitration Layer (`load_balancer/priority.py`)**:
   - Classifies requests into 4 urgency states (`SAFE`, `APPROACHING_DEADLINE`, `URGENT`, `DEADLINE_RISK`).
   - Dynamically evaluates real-time backend turnaround: $\hat{T}_{\text{comp}} = \text{latency} + \text{response\_time} \times (1 + \text{connections})$.
   - Proactively overrides ML predictions to the lowest-delay backend when deadlines are at imminent risk or when high-priority traffic contends with congested servers.
2. **Preserved Research Contract**:
   - Zero retraining required for the Phase 8 Logistic Regression model artifact.
   - 100% preservation of the 15-feature contract with zero data leakage.
   - Complete backward compatibility with standard HTTP clients (defaults safely to `NORMAL` priority and `SAFE` urgency).
3. **Observability & Header Propagation**:
   - Attaches `X-Request-Priority`, `X-Deadline-Slack`, `X-ML-Predicted-Server`, `X-Final-Backend`, `X-Priority-Override`, `X-Deadline-Override`, and `X-Routing-Reason` headers.

### Running Phase 10 Benchmarks and Analysis
```bash
# Execute Phase 10 benchmark suite across 7 scenarios x 2 algorithms x 3 repetitions
python -m benchmarks.phase10_benchmarks --repetitions 3

# Generate Phase 10 statistical analysis and research plots
python -m benchmarks.phase10_analysis
```
Comprehensive experimental report and publication-grade plots are available in [docs/priority_deadline_routing.md](docs/priority_deadline_routing.md).

---

## Phase 12: Large-Scale Dataset Expansion & Generalization Analysis

### Objective
Expand the experimental dataset by over an order of magnitude (from $N=120$ to $N=1,710$ observations across 41 experiments and 9 operational regimes) and evaluate whether the empirical machine-learning load-balancing conclusions established in Phases 6–11 generalize to unseen workload configurations and operational stress regimes.

### Core Verdict on Phase 7 Hypothesis
**Phase 7 Conclusion is WEAKENED & CONTRADICTED:**
- **Cross-Validation on Seen Data (Set A)**: **SVM** achieved the highest fidelity (**Accuracy: 96.65%, Macro F1: 92.39%**), outperforming Logistic Regression (**85.88%** Macro F1). Random Forest, Decision Tree, and XGBoost matched Logistic Regression at 91.35% accuracy and 85.88% Macro F1.
- **Out-of-Distribution Generalization (Set B)**: Under previously unseen traffic configurations, **Random Forest, Decision Tree, and XGBoost** generalized with zero degradation (**Accuracy: 94.51%, Macro F1: 89.14%**, negative generalization gap of $-3.26\%$). In contrast, **Logistic Regression suffered significant performance degradation** (**Macro F1 dropped to 71.85%**, generalization gap of $+14.03\%$), failing to accurately predict server recovery states under domain shift.
- **High-Load Transfer (Set C)**: All models achieved 100% accuracy and 1.0 Macro F1 transferring from low/medium loads to high-load regimes.
- **Traditional Baselines Evaluated Separately**: Round Robin achieved 1.0 Macro F1 on unseen configurations, while Least Connections and IP Hash exhibited severe imbalance degradation on asymmetric workloads.
- **Isolated Priority Extension**: Evaluated across 160 requests, priority-aware routing reduced P95 tail latency by 20.83% and P99 latency by 23.32% while increasing throughput by +45.36%.

### Running Phase 12 Pipeline
```bash
# Execute large-scale data collection (41 experiments, 9 operational regimes, ~1,710 requests)
python -m experiments.phase12_collector

# Run generalization benchmarking, GroupKFold CV, seen/unseen evaluation, and generate 12 research plots
python -m ml.phase12_generalization

# Run Phase 12 unit and integration tests
python -m unittest tests.test_phase12_generalization
```
Comprehensive experimental report and publication-grade plots are available in [docs/phase12_generalization.md](docs/phase12_generalization.md).

---

## Phase 13: Adaptive / Context-Aware ML Routing

### Objective
Investigate whether an adaptive model-selection layer dynamically assigning routing models based on pre-routing operational context provides measurable advantages over fixed ML routing models.

### Key Empirical Findings
- **Fixed Tree Ensembles Outperform Adaptive Selection**: Fixed **Random Forest, Decision Tree, and XGBoost** achieved top cross-validation performance (**93.44% Accuracy, 92.94% Macro F1, 6.56% suboptimal routing rate**), outperforming both **Adaptive Policy Selector** (**92.65% Accuracy, 89.53% Macro F1**) and **Adaptive Meta-Selector** (**92.65% Accuracy, 89.53% Macro F1**).
- **Selection Accuracy vs Routing Effectiveness**: Adaptive selectors achieved **92.65% model-selection accuracy**, but frequently delegating to SVM in medium and dynamic regimes bounded overall Macro F1 to 89.53%.
- **Decision Overhead**: Fixed models incurred negligible decision latency (**0.004–0.019 ms**), whereas Adaptive Policy Selector added **0.188 ms** and Adaptive Meta-Selector added **1.916 ms** (~100x overhead amplification).
- **Dynamic Switching Stability**: Transition testing across Scenarios A–D verified smooth monotonic switching tracking underlying regime shifts with zero high-frequency oscillation.
- **Fault-Tolerance & Fallback**: Automatic fallback to `LeastConnectionsRouter` guaranteed zero crashed requests upon backend outages or low model confidence.

### Running Phase 13
```bash
# Run Phase 13 meta-dataset generation, GroupKFold cross-validation, and 12 research plots
python -m ml.phase13_benchmarks

# Start live load balancer with adaptive routing
python -m load_balancer.app --port 8000 --algorithm adaptive --adaptive-strategy policy

# Run Phase 13 automated tests
python -m unittest tests.test_adaptive_routing
```
Comprehensive experimental report and publication-grade plots are available in [docs/phase13_adaptive_routing.md](docs/phase13_adaptive_routing.md).

