<h1>
  <img src="./assets/mindmemos-logo-small.png" alt="MindMemOS logo" width="40" height="40" align="absmiddle" style="vertical-align: middle;" />
  MindMemOS
</h1>

![MindMemOS Memory For AI Agents](./assets/mindmemos-hero.png)

<p align="center">
  <a href="README.md">English</a> | <a href="README_ZH.md">简体中文</a>
</p>

精准记忆用户与任务上下文，跨 Agent 迁移复用；在持续交互中演化记忆，自动沉淀 Skills，并联动文件知识系统，让经验真正成为能力。

[本地部署文档](docs/deploy/instruction_ZH.md)

## Benchmark

### Evaluation of Conversational Memory

MindMemOS-schema 在主流记忆系统竞争最激烈的 LoCoMo 基准上达到 SOTA，Overall 得分 **93.64**。

* Benchmark：[LoCoMo](https://arxiv.org/abs/2402.17753)，记忆系统最主流、竞争最激烈的基准，聚焦事实记忆与联合分析，覆盖 single-hop、multi-hop、temporal 和 open-domain 问答。
* Note：回复模型为 gpt-4.1-mini。Baseline 指标引用自 [EverMemOS](https://arxiv.org/abs/2601.02163) 论文。

| Method              | Single Hop | Multi Hop | Temporal | Open Domain | Overall   |
| :------------------ | :--------: | :-------: | :------: | :---------: | :-------: |
| MemoryOS            |    67.30   |   59.34   |  42.26   |    59.03    |   60.11   |
| Mem0                |    68.97   |   61.70   |  58.26   |    50.00    |   64.20   |
| MemU                |    74.91   |   72.34   |  43.61   |    54.17    |   66.67   |
| MemOS               |    85.37   |   79.43   |  75.08   |    64.58    |   80.76   |
| Zep                 |    90.84   |   81.91   |  77.26   |    75.00    |   85.22   |
| EverMemOS           |    96.67   |   91.84   |  89.72   |    76.04    |   93.05   |
| **MindMemOS-schema** | **97.62**  | **93.26** | 89.01 |  75.00 | **93.64** |

评测配置：[`config/mindmemos_eval/memory_evaluation_locomo.example.yaml`](config/mindmemos_eval/memory_evaluation_locomo.example.yaml)

```bash
cp config/mindmemos_eval/memory_evaluation_locomo.example.yaml config/mindmemos_eval/memory_evaluation_locomo.yaml
# 填入 API key 后执行：
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo \
  --algorithm schema \
  --manifest-output output/locomo_manifest.jsonl \
  --api-key-output config/mindmemos/api_keys.yaml
```

### Evaluation of Persona Memory

MindMemOS 通过高阶属性建模与发现，在 PersonaMem 基准上达到 SOTA，Overall 准确率领先当前 SOTA 约 **2 个百分点**。

* Benchmark：[PersonaMem](https://arxiv.org/abs/2504.14225)，以用户画像与喜好理解为中心的记忆基准，评测对用户特征的召回、追踪、重访、建议、推荐与泛化能力。
* Note：所有实验结果来源于开源代码本地运行（记忆模型、回答模型均为 gpt-4.1-mini）。

| Method              | Recall Sha. | Recall Men. (Ack. Latest) | Track Evo. | Revisit | Suggest | Recommend | Generalize | Overall          |
| :------------------ | :---------: | :-----------------------: | :--------: | :-----: | :-----: | :-------: | :--------: | :--------------: |
| MemOS               | 74.42% (96/129) | 82.35% (14/17) | 61.87% (86/139) | 77.78% (77/99) | 44.09% (41/93) | 67.27% (37/55) | 84.21% (48/57) | 67.74% (399/589) |
| EverMemOS           | 74.42% (96/129) | 64.71% (11/17) | 64.03% (89/139) | 85.86% (85/99) | 35.48% (33/93) | 65.45% (36/55) | 84.21% (48/57) | 67.57% (398/589) |
| MemU                | 64.34% (83/129) | 64.71% (11/17) | 66.20% (92/139) | 87.88% (87/99) | 31.18% (29/93) | 67.27% (37/55) | 84.21% (48/57) | 65.70% (387/589) |
| **MindMemOS-schema** | 73.64% (95/129) | **82.35%** (14/17) | **67.63%** (94/139) | 85.86% (85/99) | 35.48% (33/93) | **80.00%** (44/55) | 78.95% (45/57) | **69.61% (410/589)** |


### Evaluation of Dreaming

* Benchmark：[MemoryAgentBench](https://arxiv.org/abs/2507.05257)，多轮 Agent 记忆基准，使用 Subsequence Exact Match (SubEM) 衡量记忆增强问答效果。对比 Dreaming 前后的 SubEM 表现与记忆数量变化。
* 实验设置：top-k=50, chunk_size=1024
* Note：MIRIX 与 mem0 baseline 结果来自论文，论文配置为 chunk_size=4096。

| Pipeline | 记忆模型 | 评测模型 | 回复模型 | Single-hop SubEM | Single-hop 提升 | Single-hop 记忆数量变化 | Multi-hop SubEM | Multi-hop 提升 | Multi-hop 记忆数量变化 | 平均 SubEM | 平均提升 | 平均记忆数量变化 |
|----------|---------|---------|---------|:----------------:|:---------------:|:----------------------:|:---------------:|:--------------:|:---------------------:|:----------:|:--------:|:----------------:|
| MIRIX | GPT-4.1-mini | - | GPT-4.1-mini | 20.00% | - | - | 3.00% | - | - | 11.50% | - | - |
| mem0 | GPT-4o-mini | - | GPT-4o-mini | 18.00% | - | - | 2.00% | - | - | 10.00% | - | - |
| Ours (Vanilla) | gpt-4.1-mini | - | gpt-4.1-mini | 83.00% | | - | 10.75% | | - | 46.88% | | - |
| **Ours (Vanilla + Dreaming)** | **gpt-4.1-mini** | **-** | **gpt-4.1-mini** | **88.75%** | **+5.75%** 🟢 | **-27.5%** | **14.00%** | **+3.25%** 🟢 | **-28.3%** | **51.38%** | **+4.50%** 🟢 | **-27.9%** |

## Core Features

- **跨 Agent 可迁移**：将用户画像、偏好、项目事实、工具经验和 skill candidates 沉淀为可复用资产，让 OpenClaw、Hermes、Claude Code、OpenHands 等不同 Agent 共享或迁移同一套长期记忆。
- **记忆系统可自主演化**：通过 schema learning、dreaming、feedback 持续优化记忆质量，自动学习高频记忆点、离线巩固合并记忆，并从交互纠错中反向优化 add/search 流程。
- **记忆与 Skills 联动**：经验记忆可以沉淀为 skill candidates；skills 的执行结果、失败轨迹和用户反馈也会回流到记忆系统，推动 skills 持续演进。

## Coming Features

- **Skills 系统**：治理庞杂冗余的 skills 并智能分发；根据真实使用持续演化优化；从用户高频场景自动合成新 skills，并通过离线推演不断打磨。
- **文件系统记忆**：将散落在本地文件、文档、项目产物和 Agent 输出中的零碎知识结构化管理，构建可检索、可关联的文件知识对象或知识图谱，帮助 Agent 更好完成用户任务。
- **Agent 集成**：继续增强对代码 Agent、OpenClaw、Codex 风格工作流和长期运行多 Agent 系统的支持。

## Quickstart Guide

### 1. 本地部署

MindMemOS 使用 `uv` 管理依赖和执行本地命令。

了解详细的配置方法，可以查看 [docs/deploy/instruction_ZH.md](docs/deploy/instruction_ZH.md)。

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```

在 `config/mindmemos/api_keys.yaml` 中配置 API key 以及绑定的 `project_id`。Public memory API 使用 bearer 认证：

```text
Authorization: Bearer <api_key>
```

启动前，至少需要在 `config/mindmemos/dev.yaml` 中配置以下三类模型路由：

- `chat_model_router`：LLM 生成与抽取。
- `embed_model_router`：语义向量 embedding。
- `rerank_model_router`：检索候选结果重排。

同时确认 `database.qdrant.vector_size` 与 embedding endpoint 的输出维度一致。

启动本地服务：

```bash
make dev
```

`make dev` 会先启动全量 Docker 依赖，再启动 FastAPI。只启动核心依赖时使用：

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

Docker 档位限制：

- `make dev-core` 启动 Qdrant、Neo4j、Kafka、Kafka UI 和 kafka-exporter。
- `make dev` 和 `make db-observability` 启动全量 Docker 依赖。
- 如果 `telemetry.enabled=true`，需要使用全量依赖，确保 OTel collector 和 ClickHouse 可用；Grafana 只负责查看观测数据。

默认本地服务地址：

```text
FastAPI:   http://127.0.0.1:8000
API Docs:  http://127.0.0.1:8000/docs
```

通过 `http://127.0.0.1:8000` 调用本地 API，并使用
`config/mindmemos/api_keys.yaml` 中配置的 API key。

停止本地服务：

```bash
make dev-down
```

### 2. 调用 FastAPI

将 `config/mindmemos/api_keys.yaml` 中配置的 API key 放入
`Authorization: Bearer <api_key>` 请求头。

写入记忆：

```bash
curl -X POST http://127.0.0.1:8000/v1/memory/add \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "messages": [
      {
        "role": "user",
        "content": "I prefer iced Americano."
      }
    ],
    "mode": "sync"
  }'
```

检索记忆：

```bash
curl -X POST http://127.0.0.1:8000/v1/memory/search \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "query": "What coffee does the user prefer?",
    "top_k": 5,
    "search_strategy": "fast"
  }'
```

本地部署时，把 base URL 替换为 `http://127.0.0.1:8000` 即可。

### 3. Skills 基础使用

Skills 是带版本管理的 `SKILL.md` bundle，可以关联到 memory add 的任务轨迹，
并在后续根据真实使用持续演化。先注册一个 skill：

```bash
curl -X POST http://127.0.0.1:8000/v1/skills/register \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "travel-planner",
    "content": "# Travel Planner\n\nPrefer concise itineraries and surface hotel preferences.",
    "version_label": "v1"
  }'
```

响应中会返回 `cloud_skill_id`、`version_id` 和 `content_hash`。查看已托管的
skills：

```bash
curl -X GET http://127.0.0.1:8000/v1/skills \
  -H "Authorization: Bearer <api_key>"

curl -X POST http://127.0.0.1:8000/v1/skills/<cloud_skill_id>/get \
  -H "Authorization: Bearer <api_key>"
```

写入任务轨迹时，在 `/v1/memory/add` 中传轻量 `skill_context` 引用即可；这里
不要传完整 skill 内容：

```bash
curl -X POST http://127.0.0.1:8000/v1/memory/add \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "messages": [
      {
        "role": "user",
        "content": "Plan a two-day Guangzhou trip."
      }
    ],
    "skill_context": [
      {
        "name": "travel-planner",
        "content_hash": "<content_hash>",
        "base_version_id": "<version_id>",
        "usage": "injected"
      }
    ],
    "mode": "sync"
  }'
```

Python SDK 中同样可以通过 `MindMemOSClient.skills` 管理 skills；已注册的本地
skill 支持 push、pull、sync、update、rollback，并且在开启 skill detection 后
可以由 `client.memory.add(...)` 自动带上关联上下文。

### 4. Python SDK

Python SDK 位于当前 workspace 的 `src/mindmemos_sdk`。

先配置一次认证信息：

```bash
uv run mindmemos auth
```

之后可以直接复用本地 SDK 配置：

```python
import time

from mindmemos_sdk import MindMemOSClient
from mindmemos_sdk.memory import DialogueMessage

with MindMemOSClient() as client:
    add_result = client.memory.add(
        messages=[
            DialogueMessage(
                role="user",
                content="I prefer iced Americano.",
                timestamp=int(time.time() * 1000),
            )
        ]
    )
    for item in add_result.memories:
        print(item.operation, item.memory_id, item.content)

    search_result = client.memory.search("coffee preference", top_k=5)
    for hit in search_result.memories:
        print(hit.id, hit.memory)
```

也可以显式传入连接参数：

```python
from mindmemos_sdk import MindMemOSClient
from mindmemos_sdk.memory import DialogueMessage

client = MindMemOSClient(
    base_url="http://127.0.0.1:8000",
    api_key="mk_xxx",
    user_id="u_123",
)

result = client.memory.add(
    messages=[DialogueMessage(role="user", content="I prefer iced Americano.")]
)
print(result.memories)

client.close()
```

### 5. CLI

CLI 随 Python SDK 一起发布。

配置 CLI：

```bash
uv run mindmemos auth
```

写入和检索记忆：

```bash
uv run mindmemos memory add --content "我喜欢喝冰美式"
uv run mindmemos memory search "咖啡偏好" --top-k 5
```

管理记忆：

```bash
uv run mindmemos memory get --top-k 10
uv run mindmemos memory update <memory_id> --content "我现在更喜欢拿铁"
uv run mindmemos memory delete <memory_id>
uv run mindmemos memory feedback --text "刚才召回的偏好不准确"
uv run mindmemos memory dreaming
```

### 6. OpenClaw 插件

OpenClaw 插件位于 `plugins/openclaw-plugin`。插件会在每轮 OpenClaw
提问前检索 MindMemOS 记忆并注入相关上下文，在回合结束后通过
`mindmemos` CLI 写回完整对话。

先完成 CLI 认证：

```bash
uv run mindmemos auth
```

构建并链接本地插件，然后在 OpenClaw 中启用：

```bash
npm --prefix plugins/openclaw-plugin install
npm --prefix plugins/openclaw-plugin run build
openclaw plugins install --link /path/to/repo/plugins/openclaw-plugin
openclaw plugins enable mindmemos-memory
```

完整配置和排障说明见
[plugins/openclaw-plugin/README.md](plugins/openclaw-plugin/README.md)。

## License

MIT License.
