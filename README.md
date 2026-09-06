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
                 Client
                   │
                   │ HTTP
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

### Collection Architecture

1. **Server Endpoint (`GET /metrics`)**:
   - Exposes current runtime metrics directly in JSON format:
   ```json
   {
     "server_id": "server-1",
     "port": 8001,
     "cpu_percent": 3.2,
     "memory_percent": 0.15,
     "active_connections": 1,
     "avg_response_time_ms": 42.18,
     "network_latency_ms": 0.35,
     "request_queue_length": 0
   }
   ```
   - Bypasses application worker concurrency queues so monitoring never starves during heavy load.

2. **Metrics Collector (`monitoring/collector.py`)**:
   - `MetricsCollector`: Abstraction to query individual backends (`collect_from_server`) or all servers (`collect_all`).
   - Returns typed `ServerMetrics` dataclass instances.
   - Measures real network probe latency between collector/load balancer and backend server.
   - Graceful failure handling: unreachable or timed-out backends return `ServerMetrics(available=False, error=...)` with zeroed metric counters rather than throwing unhandled exceptions.

3. **Load Balancer Aggregation (`GET /lb-metrics`)**:
   - Central endpoint on the load balancer `:8000/lb-metrics` that polls and returns the consolidated metrics snapshot across all 3 backend servers.

### Limitations
- CPU utilization is process-specific rather than whole-system (defensible choice to isolate backend process resource consumption).
- Queue length reflects requests awaiting worker thread allocation under the bounded semaphore; kernel socket backlog is not directly exposed by Python's standard `socketserver`.
- Dataset recording and machine learning pipelines are intentionally deferred to subsequent phases.

---

## Running Tests
Run all unit and integration tests using:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Or with pytest:
```bash
python -m pytest
```
