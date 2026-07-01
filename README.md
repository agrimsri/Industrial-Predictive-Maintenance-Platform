# Industrial Predictive Maintenance Platform

A multi-modal predictive maintenance platform. This project features a real architecture with a separate ML training pipeline, a separate model-serving API, a FastAPI backend, MongoDB persistence, and (later) a frontend dashboard.

## Architecture

![Architecture Placeholder](docs/architecture.png)

* `/backend`: FastAPI app (business logic, MongoDB, orchestrates calls to serving API)
* `/ml`: ML training pipelines, notebooks/scripts, model registry
* `/serving`: Standalone model-serving API
* `/infra`: Docker setup and environment configuration
* `/docs`: Documentation and ADRs

## How to run locally

### Prerequisites
- Docker and Docker Compose

### Start the infrastructure (MongoDB)
```bash
cd infra
docker-compose up -d
```
