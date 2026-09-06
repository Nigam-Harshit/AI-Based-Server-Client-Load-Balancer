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
