# MindMemOS 部署配置说明

<p align="center">
  <a href="instruction.md">English</a> | <a href="instruction_ZH.md">简体中文</a>
</p>

本文先覆盖把服务跑起来必须关注的环境变量和 `config`。当前仓库里的 Docker Compose 主要启动依赖服务（Qdrant、Neo4j、Kafka、ClickHouse、OTel、Grafana），FastAPI 服务本身由 `make dev` / `make api` 通过 `uvicorn` 启动。

## 1. 最小启动流程

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml

# 编辑 .env 和 config/mindmemos/dev.yaml 后启动
make dev-setup
make dev
```

默认地址：

- FastAPI: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- Qdrant: `http://localhost:6333`
- Neo4j Browser: `http://localhost:7474`

`make dev` 会先启动全量 Docker 依赖，再启动 FastAPI。只启动核心依赖时使用：

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

停止本地依赖：

```bash
make dev-down
```

## 2. 必配环境变量

配置文件选择：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_CONFIG_NAME` | 选择配置名；`dev` 会读取 `config/mindmemos/dev.yaml` | `dev` |
| `MINDMEMOS_CONFIG_PATH` | 直接指定配置文件路径；设置后优先于 `MINDMEMOS_CONFIG_NAME` | 空 |

Qdrant：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_QDRANT_URL` | FastAPI 访问 Qdrant 的 HTTP 地址 | `http://localhost:6333` |
| `MINDMEMOS_QDRANT_HTTP_PORT` | Docker 暴露 Qdrant HTTP 端口 | `6333` |
| `MINDMEMOS_QDRANT_GRPC_PORT` | Docker 暴露 Qdrant gRPC 端口，也会覆盖 config 里的 `database.qdrant.grpc_port` | `6334` |
| `MINDMEMOS_QDRANT_PREFER_GRPC` | Qdrant client 是否优先使用 gRPC | `false` |
| `MINDMEMOS_QDRANT_API_KEY` | Qdrant API key；本地无鉴权可留空 | 空 |
| `MINDMEMOS_GRAFANA_QDRANT_URL` | Grafana 容器访问 Qdrant 的 HTTP 地址 | `http://qdrant:6333` |

Neo4j：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_NEO4J_URI` | FastAPI 访问 Neo4j 的 Bolt 地址 | `bolt://localhost:7687` |
| `MINDMEMOS_NEO4J_HTTP_PORT` | Docker 暴露 Neo4j Browser 端口 | `7474` |
| `MINDMEMOS_NEO4J_BOLT_PORT` | Docker 暴露 Neo4j Bolt 端口 | `7687` |
| `MINDMEMOS_NEO4J_USERNAME` | Neo4j 用户名，也是 Docker `NEO4J_AUTH` 的用户名 | `neo4j` |
| `MINDMEMOS_NEO4J_PASSWORD` | Neo4j 密码，也是 Docker `NEO4J_AUTH` 的密码 | `mindmemos_dev_password` |

可选依赖：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS` | Kafka 地址；只有 config 里 `kafka.enabled=true` 时服务才会启动消费者/生产者 | `localhost:9092` |
| `MINDMEMOS_TELEMETRY_ENDPOINT` | OTel HTTP endpoint；只有 config 里 `telemetry.enabled=true` 时会上报 | `http://localhost:4318` |
| `MINDMEMOS_CLICKHOUSE_USER` / `MINDMEMOS_CLICKHOUSE_PASSWORD` / `MINDMEMOS_CLICKHOUSE_DB` | ClickHouse/Grafana 观测数据配置 | 见 `.env.example` |

API 监听地址：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_API_HOST` | `make dev` / `make api` 启动 FastAPI 的 host | `127.0.0.1` |
| `MINDMEMOS_API_PORT` | `make dev` / `make api` 启动 FastAPI 的 port | `8000` |

## 3. Docker 相关

本地依赖通过：

```bash
docker compose --env-file .env -f dockers/docker-compose.memory.yml up -d --wait qdrant neo4j kafka kafka-ui kafka-exporter
```

`make dev-core` 会启动 Qdrant、Neo4j、Kafka、Kafka UI 和 kafka-exporter。`make dev` 会先启动全量 Docker 依赖，再启动 FastAPI。`make db` 仍保留为全量依赖的兼容入口，等同于 `make db-observability`。

Docker Compose 内的核心服务：

- `qdrant`: 存 memory/entity/source 向量和 payload。
- `neo4j`: 存图关系。
- `kafka`: 异步任务队列；默认 config 里未开启也可以先跑着。
- `clickhouse` + `otel-collector` + `grafana`: 观测链路；不需要观测时可以在 config 里关掉 `telemetry.enabled`。

本地部署时，`.env` 里的端口变量要和 `config/mindmemos/dev.yaml` 里的连接地址对齐。代码启动时还会用环境变量覆盖这些连接字段：

- `database.qdrant.url`
- `database.qdrant.api_key`
- `database.qdrant.grpc_port`
- `database.qdrant.prefer_grpc`
- `database.neo4j.uri`
- `database.neo4j.username`
- `database.neo4j.password`
- `kafka.bootstrap_servers`
- `telemetry.telemetry_endpoint`

## 4. LLM 配置

LLM 用于记忆抽取、schema 处理、dreaming 等生成任务。需要配置 `chat_model_router`：

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

注意：

- `model` 是 LiteLLM 风格的模型名，OpenAI 兼容接口通常写成 `openai/<model-name>`。
- `api_base` 要包含 `/v1`，除非你的供应商文档明确不是这个形式。
- `api_key` 不要提交到仓库；本地写在未提交的 `config/mindmemos/dev.yaml` 即可。
- 可以配置多个 endpoint，router 会按 `routing_strategy` 路由。

## 5. Embedding 配置

Embedding 是必须配置的；服务启动时会校验 embedding 输出维度和 Qdrant 向量维度是否一致。

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

重点：

- `database.qdrant.vector_size` 必须等于 embedding 模型实际输出维度。
- 如果 embedding 模型支持自定义维度，`dimensions` 和 `vector_size` 也要一致。
- Qdrant collection 已经用旧维度创建后，单纯改 `vector_size` 不会自动迁移旧 collection；本地开发可以 `make db-clean` 清掉 volume 后重建。

## 6. Rerank 配置（可选）

Rerank 用于检索候选结果重排，提升搜索精度，但不是服务启动的硬依赖。没有外部 rerank endpoint 时，基础 add/search 仍可运行；代码会使用现有召回结果或 fallback 逻辑。

需要外部 reranker 时配置：

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

不使用外部 reranker 时可以写成：

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

强调：`rerank` 是可选增强项。生产环境建议先把 Docker、LLM、Embedding 跑稳，再接入 rerank。

## 7. 认证配置

本地默认使用 API key：

```yaml
auth:
  mode: api_key
  api_key_file: api_keys.yaml
```

`api_key_file` 是相对 config 文件目录解析的，所以默认指向 `config/mindmemos/api_keys.yaml`。本地示例里已有：

- `dev-api-key-001`: vanilla memory
- `dev-api-key-002`: schema memory

调用 API 时使用：

```text
Authorization: Bearer <api_key>
```

## 8. 最小检查清单

启动前至少确认：

- `.env` 里的 Qdrant、Neo4j 端口没有和本机已有服务冲突。
- `config/mindmemos/dev.yaml` 存在。
- `chat_model_router.endpoints[0].api_key` / `api_base` / `model` 可用。
- `embed_model_router.endpoints[0].api_key` / `api_base` / `model` 可用。
- `database.qdrant.vector_size` 等于 embedding 输出维度。
- 不需要 rerank 时，`rerank_model_router.endpoints` 可以留空，并关闭相关 `use_reranker`。
