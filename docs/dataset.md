# Experimental Dataset Specification

## 1. Overview
This document specifies the schema, collection methodology, labeling policy, and validation rules for experimental datasets collected in the **AI-Based Server-Client Load Balancer** research project.

The dataset captures empirical measurements across three identical HTTP backend servers under controlled workloads routed by traditional algorithms (`round_robin`, `least_connections`, `ip_hash`).

---

## 2. Dataset Schema

Every observation represents a single routing decision. The features represent the measured state of all three servers **strictly before** the routing decision is made.

| Column Name | Type | Unit / Range | Temporal Stage | Description |
| :--- | :--- | :--- | :--- | :--- |
| `experiment_id` | `str` | Text ID | Setup | Unique identifier for the experimental run. |
| `timestamp` | `float` | Epoch seconds | Pre-Routing | Timestamp captured immediately before routing. |
| `request_id` | `int` | `1, 2, ...` | Request | Sequence number of the request within the experiment. |
| `workload_scenario` | `str` | Category | Setup | Scenario name (`low_traffic`, `burst_traffic`, etc.). |
| `request_type` | `str` | Category | Request | Endpoint type (`process`, `health`). |
| `request_size` | `int` | Bytes (`>= 0`) | Request | Payload size of client request. |
| `concurrency` | `int` | Count (`>= 1`) | Setup | Configured concurrency level of the workload generator. |
| `request_rate` | `float` | req/s (`> 0` or null) | Setup | Target request rate (null if unthrottled). |
| `routing_algorithm` | `str` | Category | Setup | Algorithm used (`round_robin`, `least_connections`, `ip_hash`). |
| `server_1_cpu` | `float` | `%` (`0.0` - `100.0`) | Pre-Routing | CPU utilization of Server 1 before routing. |
| `server_1_memory` | `float` | `%` (`0.0` - `100.0`) | Pre-Routing | Memory utilization of Server 1 before routing. |
| `server_1_connections`| `int` | Count (`>= 0`) | Pre-Routing | Active connections on Server 1 before routing. |
| `server_1_response_time`| `float` | `ms` (`>= 0.0`) | Pre-Routing | Recent average response time of Server 1. |
| `server_1_network_latency`| `float`| `ms` (`>= 0.0`) | Pre-Routing | Network probe round-trip latency to Server 1. |
| `server_1_queue_length`| `int` | Count (`>= 0`) | Pre-Routing | Requests queued on Server 1 before routing. |
| `server_2_cpu` | `float` | `%` (`0.0` - `100.0`) | Pre-Routing | CPU utilization of Server 2 before routing. |
| `server_2_memory` | `float` | `%` (`0.0` - `100.0`) | Pre-Routing | Memory utilization of Server 2 before routing. |
| `server_2_connections`| `int` | Count (`>= 0`) | Pre-Routing | Active connections on Server 2 before routing. |
| `server_2_response_time`| `float` | `ms` (`>= 0.0`) | Pre-Routing | Recent average response time of Server 2. |
| `server_2_network_latency`| `float`| `ms` (`>= 0.0`) | Pre-Routing | Network probe round-trip latency to Server 2. |
| `server_2_queue_length`| `int` | Count (`>= 0`) | Pre-Routing | Requests queued on Server 2 before routing. |
| `server_3_cpu` | `float` | `%` (`0.0` - `100.0`) | Pre-Routing | CPU utilization of Server 3 before routing. |
| `server_3_memory` | `float` | `%` (`0.0` - `100.0`) | Pre-Routing | Memory utilization of Server 3 before routing. |
| `server_3_connections`| `int` | Count (`>= 0`) | Pre-Routing | Active connections on Server 3 before routing. |
| `server_3_response_time`| `float` | `ms` (`>= 0.0`) | Pre-Routing | Recent average response time of Server 3. |
| `server_3_network_latency`| `float`| `ms` (`>= 0.0`) | Pre-Routing | Network probe round-trip latency to Server 3. |
| `server_3_queue_length`| `int` | Count (`>= 0`) | Pre-Routing | Requests queued on Server 3 before routing. |
| `selected_server` | `str` | URL / Server ID | Post-Routing | Backend chosen by the baseline load balancer. |
| `actual_response_time`| `float` | `ms` (`>= 0.0`) | Post-Routing | Measured duration of the routed request. |
| `request_success` | `bool` | `True` / `False` | Post-Routing | Whether the request completed successfully (2xx/3xx). |
| `best_server` | `str` | URL / Server ID / null | Label | Empirically determined optimal server target. |

---

## 3. Critical Temporal Separation (No Future Leakage)

To prevent data leakage in future ML training:
1. **Pre-Routing Features**: All 18 server state metrics (`server_1_*`, `server_2_*`, `server_3_*`) are captured **before** the routing decision is made.
2. **Outcome Metrics**: `actual_response_time`, `request_success`, and `best_server` are populated **only after** request completion.
3. Feature vectors strictly contain information that a real-time router would possess when selecting a target server.

---

## 4. Label Generation Strategy

The target label `best_server` represents the best backend outcome **independently of which server the load balancer selected**.

### Strategy:
1. **Failure Case**: If the load balancer routed to a server that failed (`request_success == False`), that server is eliminated from candidate selection.
2. **Empirical Performance Proxy**:
   Each server $S_i$ is evaluated based on its queue-aware expected completion latency:
   $$\text{Cost}(S_i) = \text{network\_latency}_i + \text{response\_time}_i \times (1 + \text{connections}_i + \text{queue\_length}_i)$$
   - The server with the lowest $\text{Cost}(S_i)$ is designated `best_server`.
3. **Ambiguity / Tie Policy**:
   - When all candidate servers exhibit nearly identical cost (e.g. within $\le 2\text{ms}$ on an idle cluster with 0 connections), there is no empirically measurable performance advantage.
   - In this situation, the label is explicitly set to `None` / `unassigned` rather than arbitrarily picking a server.
   - During future ML processing, observations with unassigned labels may be analyzed separately or excluded from classification training to avoid injecting arbitrary noise.

---

## 5. Experimental Repetition Strategy

1. **Scenarios**: All 7 workload scenarios (`low_traffic`, `medium_traffic`, `high_traffic`, `burst_traffic`, `cpu_heavy`, `mixed`, `dynamic`) are executed.
2. **Algorithms**: Each scenario is evaluated under `round_robin`, `least_connections`, and `ip_hash`.
3. **Repetition**: Multiple independent runs are executed per combination with fixed random seeds (`seed=42, 43, ...`) for reproducibility.
4. **Metadata**: Each experiment generates a `metadata_<id>.json` tracking software versions, timestamps, and parameters.

---

## 6. Data Quality and Validation Rules

Validation routines verify:
- Complete column presence and schema adherence.
- Monotonic timestamp sequencing (`timestamp <= request_start <= request_end`).
- Absence of unexpected null values in primary features.
- Value bounds (`cpu_percent` $\in [0, 100]$, `memory_percent` $\in [0, 100]$, counts $\ge 0$, latencies $\ge 0$).
- Request ID uniqueness within each experiment.
- Label distribution summary across candidate servers.
