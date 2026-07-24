# MindMemOS Deployment Configuration Guide

<p align="center">
  <a href="instruction.md">English</a> | <a href="instruction_ZH.md">简体中文</a>
</p>

This document covers the environment variables and `config` settings required to run the service locally. Docker Compose in this repository mainly starts dependency services such as Qdrant, Neo4j, Kafka, ClickHouse, OTel, and Grafana. The FastAPI service itself is started by `make dev` / `make api` through `uvicorn`.

## 1. Minimal Startup Flow

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml

# Edit .env and config/mindmemos/dev.yaml, then start the service.
make dev-setup
make dev
```

Default local addresses:

- FastAPI: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- Qdrant: `http://localhost:6333`
- Neo4j Browser: `http://localhost:7474`

`make dev` starts the full Docker dependency stack first, then starts FastAPI. To start only core dependencies:

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

Stop local dependencies:

```bash
make dev-down
```

## 2. Required Environment Variables

Config selection:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_CONFIG_NAME` | Selects the config name; `dev` reads `config/mindmemos/dev.yaml`. | `dev` |
| `MINDMEMOS_CONFIG_PATH` | Direct config file path; takes precedence over `MINDMEMOS_CONFIG_NAME` when set. | Empty |

Qdrant:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_QDRANT_URL` | HTTP address used by FastAPI to access Qdrant. | `http://localhost:6333` |
| `MINDMEMOS_QDRANT_HTTP_PORT` | Qdrant HTTP port exposed by Docker. | `6333` |
| `MINDMEMOS_QDRANT_GRPC_PORT` | Qdrant gRPC port exposed by Docker; also overrides `database.qdrant.grpc_port` in config. | `6334` |
| `MINDMEMOS_QDRANT_PREFER_GRPC` | Whether the Qdrant client prefers gRPC. | `false` |
| `MINDMEMOS_QDRANT_API_KEY` | Qdrant API key; can be empty for unauthenticated local setup. | Empty |
| `MINDMEMOS_GRAFANA_QDRANT_URL` | HTTP address used by the Grafana container to access Qdrant. | `http://qdrant:6333` |

Neo4j:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_NEO4J_URI` | Bolt address used by FastAPI to access Neo4j. | `bolt://localhost:7687` |
| `MINDMEMOS_NEO4J_HTTP_PORT` | Neo4j Browser port exposed by Docker. | `7474` |
| `MINDMEMOS_NEO4J_BOLT_PORT` | Neo4j Bolt port exposed by Docker. | `7687` |
| `MINDMEMOS_NEO4J_USERNAME` | Neo4j username; also used in Docker `NEO4J_AUTH`. | `neo4j` |
| `MINDMEMOS_NEO4J_PASSWORD` | Neo4j password; also used in Docker `NEO4J_AUTH`. | `mindmemos_dev_password` |

Optional dependencies:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS` | Kafka address; consumers/producers start only when `kafka.enabled=true` in config. | `localhost:9092` |
| `MINDMEMOS_TELEMETRY_ENDPOINT` | OTel HTTP endpoint; used only when `telemetry.enabled=true` in config. | `http://localhost:4318` |
| `MINDMEMOS_OTEL_GRPC_ENDPOINT` / `MINDMEMOS_OTEL_HTTP_ENDPOINT` | In-container listen addresses for the OTel Collector. | `0.0.0.0:4317` / `0.0.0.0:4318` |
| `MINDMEMOS_OTEL_KAFKA_EXPORTER_TARGET` | In-container kafka-exporter scrape target for the OTel Collector. | `kafka-exporter:9308` |
| `MINDMEMOS_OTEL_CLICKHOUSE_ENDPOINT` | In-container ClickHouse native endpoint used by the OTel Collector. | `tcp://clickhouse:9000?dial_timeout=10s` |
| `MINDMEMOS_CLICKHOUSE_USER` / `MINDMEMOS_CLICKHOUSE_PASSWORD` / `MINDMEMOS_CLICKHOUSE_DB` | ClickHouse/Grafana observability configuration. | See `.env.example` |

API bind address:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_API_HOST` | Host used by `make dev` / `make api` to start FastAPI. | `127.0.0.1` |
| `MINDMEMOS_API_PORT` | Port used by `make dev` / `make api` to start FastAPI. | `8000` |

## 3. Docker

Local dependencies are started with:

```bash
docker compose --env-file .env -f dockers/docker-compose.memory.yml up -d --wait qdrant neo4j kafka kafka-ui kafka-exporter
```

`make dev-core` runs Qdrant, Neo4j, Kafka, Kafka UI, and kafka-exporter. `make dev` runs the full Docker dependency stack first, then starts FastAPI. `make db` remains as a compatibility entry point for the full dependency tier and is equivalent to `make db-observability`.

Core services in Docker Compose:

- `qdrant`: stores memory/entity/source vectors and payloads.
- `neo4j`: stores graph relationships.
- `kafka`: async task queue; it can run even when the default config does not enable it.
- `clickhouse` + `otel-collector` + `grafana`: observability stack; disable `telemetry.enabled` in config when observability is not needed.

For local deployment, port variables in `.env` must match connection addresses in `config/mindmemos/dev.yaml`. At startup, environment variables also override these config fields:

- `database.qdrant.url`
- `database.qdrant.api_key`
- `database.qdrant.grpc_port`
- `database.qdrant.prefer_grpc`
- `database.neo4j.uri`
- `database.neo4j.username`
- `database.neo4j.password`
- `kafka.bootstrap_servers`
- `telemetry.telemetry_endpoint`

## 4. LLM Configuration

LLMs are used for memory extraction, schema processing, dreaming, and other generation tasks. Configure `chat_model_router`:

```yaml
chat_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/gpt-4.1-mini
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 1200
      temperature: 0.0
      num_retries: 3
      extra_body: {}
```

Notes:

- `model` uses LiteLLM-style model names. OpenAI-compatible endpoints usually use `openai/<model-name>`.
- `api_base` should include `/v1`, unless your provider explicitly documents a different format.
- Do not commit `api_key`; keep it in the untracked local `config/mindmemos/dev.yaml`.
- Multiple endpoints can be configured, and the router dispatches by `routing_strategy`.

## 5. Embedding Configuration

Embedding is required. On startup, the service validates that the embedding output dimension matches the Qdrant vector dimension.

```yaml
embed_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/qwen3-embedding-4b
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 600
      num_retries: 3
      dimensions: 2560
      extra_body: {}

database:
  qdrant:
    vector_size: 2560
    semantic_vector_name: semantic
    bm25_vector_name: bm25
```

Key points:

- `database.qdrant.vector_size` must equal the actual output dimension of the embedding model.
- If the embedding model supports custom dimensions, `dimensions` and `vector_size` must also match.
- If Qdrant collections were already created with an old dimension, changing `vector_size` alone will not migrate them. For local development, run `make db-clean` to clear volumes and rebuild.

## 6. Rerank Configuration (Optional)

Rerank improves search precision by reranking retrieval candidates, but it is not required for service startup. Without an external rerank endpoint, basic add/search still works; the code uses existing recall results or fallback logic.

Configure an external reranker when needed:

```yaml
rerank_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/qwen3-reranker-4b
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 600
      num_retries: 3

algo_config:
  search:
    rerank:
      enabled: true
      max_query_length: 100
      max_doc_length: 5000
      max_batch_size: 20
      max_concurrent_batches: 1
      request_timeout: 5.0
    vanilla:
      use_reranker: true
    schema_search:
      entity:
        use_reranker: true
```

When not using an external reranker:

```yaml
rerank_model_router:
  routing_strategy: simple-shuffle
  endpoints: []

algo_config:
  search:
    rerank:
      enabled: false
    vanilla:
      use_reranker: false
    schema_search:
      entity:
        use_reranker: false
```

`rerank` is an optional enhancement. For production, stabilize Docker, LLM, and Embedding first, then add rerank.

## 7. Authentication Configuration

Local setup uses API keys by default:

```yaml
auth:
  mode: api_key
  api_key_file: api_keys.yaml
```

`api_key_file` is resolved relative to the config file directory, so by default it points to `config/mindmemos/api_keys.yaml`. The local example includes:

- `dev-api-key-001`: vanilla memory
- `dev-api-key-002`: schema memory

Use this header when calling APIs:

```text
Authorization: Bearer <api_key>
```

## 8. Minimal Checklist

Before starting, check at least:

- Qdrant and Neo4j ports in `.env` do not conflict with existing local services.
- `config/mindmemos/dev.yaml` exists.
- `chat_model_router.endpoints[0].api_key` / `api_base` / `model` are valid.
- `embed_model_router.endpoints[0].api_key` / `api_base` / `model` are valid.
- `database.qdrant.vector_size` equals the embedding output dimension.
- If rerank is not needed, `rerank_model_router.endpoints` can stay empty, and related `use_reranker` flags should be disabled.
