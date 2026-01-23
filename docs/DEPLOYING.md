# NVIDIA NIM RAG Demo - Deployment Guide

## Deployment Options

| Option | Use Case | Cost |
|--------|----------|------|
| Local Docker | Development | Free |
| Cloud VM | Production | $50-200/mo |
| Kubernetes | Enterprise | Varies |

---

## Local Docker Deployment

### 1. Build Images

```bash
# Build application image
docker build -t nvidia-nim-rag-demo .

# Or use docker-compose
docker-compose build
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### 3. Access Application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API Docs | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/ragdb
      - NVIDIA_API_KEY=${NVIDIA_API_KEY}
    depends_on:
      - postgres

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://app:8000

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: ragdb
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

---

## Cloud Deployment (AWS)

### 1. Prerequisites

```bash
# AWS CLI configured
aws configure

# ECS CLI (optional)
brew install amazon-ecs-cli
```

### 2. Push to ECR

```bash
# Create repository
aws ecr create-repository --repository-name nvidia-nim-rag-demo

# Login to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

# Tag and push
docker tag nvidia-nim-rag-demo:latest <account>.dkr.ecr.<region>.amazonaws.com/nvidia-nim-rag-demo:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/nvidia-nim-rag-demo:latest
```

### 3. Deploy to ECS

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name rag-cluster

# Create task definition (see task-definition.json)
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster rag-cluster \
  --service-name rag-service \
  --task-definition rag-task:1 \
  --desired-count 2
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | From build.nvidia.com |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `LOG_LEVEL` | No | DEBUG, INFO, WARNING, ERROR |
| `CORS_ORIGINS` | No | Allowed origins for CORS |

---

## CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push
        run: |
          docker build -t $ECR_REGISTRY/nvidia-nim-rag-demo:${{ github.sha }} .
          docker push $ECR_REGISTRY/nvidia-nim-rag-demo:${{ github.sha }}

      - name: Update ECS
        run: |
          aws ecs update-service \
            --cluster rag-cluster \
            --service rag-service \
            --force-new-deployment
```

---

## Monitoring

### Health Check Endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": await check_db(),
        "nim_api": await check_nim()
    }
```

### Metrics Endpoint

```python
@app.get("/metrics")
async def metrics():
    return {
        "requests_total": counter.value,
        "latency_p50": histogram.p50,
        "latency_p99": histogram.p99,
        "active_connections": gauge.value
    }
```

### Logging

```python
import logging

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

---

## Scaling Considerations

| Metric | Threshold | Action |
|--------|-----------|--------|
| CPU > 80% | Scale up | Add instances |
| Latency > 2s | Investigate | Check NIM API |
| Error rate > 1% | Alert | Review logs |

---

*Created: 2026-01-22 | Author: Waseem Habib | QbitLoop*
