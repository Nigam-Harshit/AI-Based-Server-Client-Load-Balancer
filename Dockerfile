# Production-Oriented Multi-Service Dockerfile for AI-Based Load Balancer
FROM python:3.12-slim

# Security & Optimization Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal OS dependencies for healthchecks and monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache layer for Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy required application modules and candidate model artifacts
COPY config /app/config
COPY load_balancer /app/load_balancer
COPY monitoring /app/monitoring
COPY server /app/server
COPY client /app/client
COPY ml /app/ml
COPY models /app/models

# Create non-root user for security hardening
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose potential service ports
EXPOSE 8000 8001 8002 8003

# Default to starting the Load Balancer
CMD ["python", "-m", "load_balancer.app"]

