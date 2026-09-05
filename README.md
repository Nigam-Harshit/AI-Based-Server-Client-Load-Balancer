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

