# Phase 8: Machine Learning Driven Load Balancer Integration

## 1. Overview & Architecture

Phase 8 integrates the trained and evaluated ML classification model directly into the live central HTTP Load Balancer (`load_balancer/app.py` & `load_balancer/router.py`).

### Routing Request Flow
```text
Client Request
      │
      ▼
Load Balancer (:8000)
      │
      ▼
MetricsCollector.collect_all() (Real-time probe across :8001, :8002, :8003)
      │
      ▼
ml.features.extract_features_from_metrics()
      │ (15 active pre-routing features matching training contract)
      ▼
MLRouter.select()
      │
      ├── Model Prediction (Pipeline: StandardScaler + LogisticRegression)
      ├── Probability Distribution (predict_proba -> confidence score)
      │
      ▼
Health & Availability Validation
      ├── Target healthy? ──► Route to Predicted Backend Server
      └── Target down/error? ──► Fallback to LeastConnectionsRouter
      │
      ▼
Forward HTTP Request to Backend Node (:8001 / :8002 / :8003)
      │
      ▼
Response to Client + Observability Headers:
      - X-Backend-Server: http://127.0.0.1:800x
      - X-ML-Predicted-Server: http://127.0.0.1:800x
      - X-ML-Confidence: 0.9106
      - X-ML-Fallback: false | true
```

---

## 2. Pluggable ML Model Design

The routing architecture is decoupled from any specific algorithm:
- **Primary Production Model**: `LogisticRegression` (with `StandardScaler` in an scikit-learn `Pipeline`), persisted at `models/logistic_regression.joblib`.
- **Interchangeability**: Any alternative estimator or pipeline that implements `.predict()` and optional `.predict_proba()` (such as `RandomForestClassifier` at `models/random_forest.joblib` or `XGBClassifier`) can be passed via the `--model-path` CLI flag or injected programmatically into `MLRouter(backends=..., model_path=...)`.
- **Model Metadata**: Stored alongside artifacts in `models/metadata.json`, capturing feature ordering, target labels, training sample counts, and random seed.

---

## 3. Strict 15 Pre-Routing Feature Contract

Feature extraction is centralized in `ml.features.extract_features_from_metrics()` and matches the schema validated in Phases 6 and 7:

| Feature Name | Server | Description |
| :--- | :--- | :--- |
| `server_1_cpu` | Server 1 | CPU utilization (%) |
| `server_1_memory` | Server 1 | Memory utilization (%) |
| `server_1_connections` | Server 1 | Active connections |
| `server_1_response_time` | Server 1 | Average response time (ms) |
| `server_1_network_latency` | Server 1 | Round-trip probe latency (ms) |
| `server_2_cpu` | Server 2 | CPU utilization (%) |
| `server_2_memory` | Server 2 | Memory utilization (%) |
| `server_2_connections` | Server 2 | Active connections |
| `server_2_response_time` | Server 2 | Average response time (ms) |
| `server_2_network_latency` | Server 2 | Round-trip probe latency (ms) |
| `server_3_cpu` | Server 3 | CPU utilization (%) |
| `server_3_memory` | Server 3 | Memory utilization (%) |
| `server_3_connections` | Server 3 | Active connections |
| `server_3_response_time` | Server 3 | Average response time (ms) |
| `server_3_network_latency` | Server 3 | Round-trip probe latency (ms) |

*Zero-variance queue length features (`server_x_queue_length`) are strictly omitted, and outcome metrics (`actual_response_time`, `request_success`) are never accessed during routing.*

---

## 4. Safety Guarantees & Fallback Policy

The system enforces fail-safe routing under all degradation scenarios:

1. **Unreachable Backend Detection**: During metric collection, each backend's `/metrics` probe measures responsiveness. If a backend fails or times out, its `available` flag is set to `False`.
2. **Offline Target Protection**: If the ML model selects a backend that is currently down or unhealthy, the decision is immediately overridden.
3. **Graceful Fallback**: The request automatically delegates to `LeastConnectionsRouter` among the remaining healthy backend nodes.
4. **Exception Handling**: If model inference raises an exception, the model file is missing, or feature extraction fails, the load balancer logs the incident, sets `X-ML-Fallback: true`, and safely forwards the request via `LeastConnectionsRouter`.
5. **No Broken Requests**: Requests are never forwarded to a confirmed dead server due to an outdated or misconfigured model prediction.

---

## 5. Observability & Monitoring

Every routed response includes HTTP inspection headers:
- `X-Backend-Server`: The actual backend server URL handling the request.
- `X-ML-Predicted-Server`: The backend server URL predicted by the ML model.
- `X-ML-Confidence`: The probability score of the predicted class (formatted to 4 decimal places).
- `X-ML-Fallback`: Boolean flag (`true` or `false`) indicating whether fallback routing intervened.

Detailed logs are output per request:
```text
[ml] client=127.0.0.1 backend=http://127.0.0.1:8001 path=/health status=200 duration=0.0051s ml_pred=http://127.0.0.1:8001 conf=0.9106 fallback=False
```

---

## 6. CLI Usage & Switching Algorithms

### Running Traditional Baseline Algorithms
```bash
# Round Robin
python -m load_balancer.app --port 8000 --algorithm round_robin

# Least Connections
python -m load_balancer.app --port 8000 --algorithm least_connections

# IP Hash
python -m load_balancer.app --port 8000 --algorithm ip_hash
```

### Running ML-Driven Routing
```bash
# Default (Logistic Regression model)
python -m load_balancer.app --port 8000 --algorithm ml

# Custom model artifact (e.g. Random Forest)
python -m load_balancer.app --port 8000 --algorithm ml --model-path models/random_forest.joblib
```

### Retraining Model Artifacts
```bash
# Retrain Logistic Regression model
python -m ml.trainer

# Retrain Random Forest model
python -c "from ml.trainer import train_and_persist_model; train_and_persist_model(model_type='RandomForest')"
```
