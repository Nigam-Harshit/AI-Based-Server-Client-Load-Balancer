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

## Phase 2: Load Balancer Implementation

### Architecture
```text
                 Client
                   │
                   │ HTTP
                   ▼
           ┌───────────────┐
           │ Load Balancer │
           │     :8000     │
           └───────┬───────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    Server 1   Server 2   Server 3
     :8001      :8002      :8003
```

### Components

1. **Backend Server Nodes (`server/app.py`)**:
   - Lightweight, reusable HTTP server instances implemented with Python standard library `ThreadingHTTPServer`.
   - Endpoints:
     - `GET /health`: Liveness probe returning server ID, status, and port.
     - `GET /process?duration=...`: Simulated computational workload returning completion status and duration.
   - Configurable via CLI arguments (`--id`, `--host`, `--port`).

2. **Backend Configuration (`config/backends.py`)**:
   - Centralized list of backend URLs (`DEFAULT_BACKENDS = ["http://127.0.0.1:8001", "http://127.0.0.1:8002", "http://127.0.0.1:8003"]`).

3. **Routing Abstraction (`load_balancer/router.py`)**:
   - `BaseRouter`: Abstract base class providing `select(client_ip)` and `release(backend, success)` lifecycle hooks, along with a `route(client_ip)` context manager guaranteeing cleanup even on failures.
   - **Round Robin (`RoundRobinRouter`)**:
     - Sequentially cycles through configured backend servers.
     - Thread-safe synchronization via `threading.Lock`.
   - **Least Connections (`LeastConnectionsRouter`)**:
     - Routes requests to the backend server with the lowest number of currently active connections.
     - **Tie-Breaking Policy**: When multiple servers share the same minimum connection count, the server appearing earliest in the configured backend list (lowest index) is chosen deterministically.
     - Thread-safe tracking with increment on `select()` and decrement on `release()`.
     - *Clarification*: Connection counts here represent strictly internal load balancer routing state for decision making and do NOT represent the Phase 3 real-time server monitoring metrics.
   - **IP Hash (`IPHashRouter`)**:
     - Deterministically maps client IP addresses to servers using `hashlib.md5`.
     - Formula: `int(hashlib.md5(ip.encode("utf-8")).hexdigest(), 16) % len(backends)`.
     - Falls back to `127.0.0.1` if client IP is missing.
   - Router Factory: `get_router(algorithm, backends)`.

4. **Central Load Balancer Server (`load_balancer/app.py`)**:
   - Listens on configurable host and port (default `:8000`).
   - Accepts client HTTP requests, selects a backend using the configured router, and transparently forwards requests.
   - Adds `X-Backend-Server` response header identifying the servicing backend.
   - Propagates client IP via `X-Forwarded-For`.
   - Robust error handling:
     - Returns `502 Bad Gateway` on connection refused or backend unavailability.
     - Returns `504 Gateway Timeout` on backend request timeouts.
   - Structured logging of routing events (algorithm, client IP, selected backend, path, status, duration).

### Current Limitations
- Health checks are passive; dead backends are not yet automatically removed from routing rotation.
- Real-time CPU, memory, and latency metric collection is deferred to Phase 3.
- Machine learning models and workload generators are deferred to later phases.

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
