# Experimental Data Repository

## Overview
This directory stores experimental datasets collected from real-time measurements across the 3 HTTP backend servers under controlled workload scenarios.

## Directory Structure
```text
data/
├── raw/            # Immutable raw experimental observations and metadata
├── processed/      # Cleaned, validated datasets with generated labels
└── README.md       # Data repository documentation
```

## Raw vs. Processed Data Policy
1. **Immutability of Raw Data (`data/raw/`)**:
   - Every experimental run writes an immutable CSV observation file (`experiment_<id>.csv`) and a metadata file (`metadata_<id>.json`).
   - Raw measurements are never overwritten, edited, or fabricated.
   - Raw files preserve exact measured values, raw timestamps, and failure conditions.

2. **Processed Data (`data/processed/`)**:
   - Contains cleaned and validated datasets (`processed_<id>.csv` or aggregated `dataset_combined.csv`).
   - Target labels (`best_server`) are populated according to the documented labeling strategy.
   - Ambiguous or tied observations are marked with explicit labels or documented exclusion flags.

## Schema Overview
Each observation row represents a single routing decision and records the complete cluster state before routing:
- **Identifiers**: `experiment_id`, `timestamp`, `request_id`
- **Workload Metadata**: `workload_scenario`, `request_type`, `request_size`, `concurrency`, `request_rate`
- **Routing**: `routing_algorithm`, `selected_server`
- **Server 1 State (Pre-Routing)**: `server_1_cpu`, `server_1_memory`, `server_1_connections`, `server_1_response_time`, `server_1_network_latency`, `server_1_queue_length`
- **Server 2 State (Pre-Routing)**: `server_2_cpu`, `server_2_memory`, `server_2_connections`, `server_2_response_time`, `server_2_network_latency`, `server_2_queue_length`
- **Server 3 State (Pre-Routing)**: `server_3_cpu`, `server_3_memory`, `server_3_connections`, `server_3_response_time`, `server_3_network_latency`, `server_3_queue_length`
- **Outcome**: `actual_response_time`, `request_success`
- **Target Label**: `best_server`
