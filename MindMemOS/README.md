<h1>
  <img src="./assets/mindmemos-logo-small.png" alt="MindMemOS logo" width="40" height="40" align="absmiddle" style="vertical-align: middle;" />
  MindMemOS
</h1>

![MindMemOS Memory For AI Agents](./assets/mindmemos-hero.png)


<p align="center">
  <a href="README.md">English</a> | <a href="README_ZH.md">简体中文</a>
</p>

MindMemOS is an open-source long-term memory system for AI agents and applications. It helps agents turn conversations, files, tool traces, feedback, and offline reflection into searchable, updateable, project-isolated memory.

[Local deployment documentation](docs/deploy/instruction.md)

## Benchmark

### Evaluation of Conversational Memory

MindMemOS-schema achieves state-of-the-art on LoCoMo, the most competitive benchmark for long-term memory systems, with an overall score of **93.64**.

* Benchmark: [LoCoMo](https://arxiv.org/abs/2402.17753), the most mainstream and fiercely contested benchmark for long-term memory systems, focused on factual memory retention and joint analysis, covering single-hop, multi-hop, temporal, and open-domain question answering.
* Note: Answer model: gpt-4.1-mini. Baseline metrics are cited from the [EverMemOS](https://arxiv.org/abs/2601.02163) paper.

| Method              | Single Hop | Multi Hop | Temporal | Open Domain | Overall   |
| :------------------ | :--------: | :-------: | :------: | :---------: | :-------: |
| MemoryOS            |    67.30   |   59.34   |  42.26   |    59.03    |   60.11   |
| Mem0                |    68.97   |   61.70   |  58.26   |    50.00    |   64.20   |
| MemU                |    74.91   |   72.34   |  43.61   |    54.17    |   66.67   |
| MemOS               |    85.37   |   79.43   |  75.08   |    64.58    |   80.76   |
| Zep                 |    90.84   |   81.91   |  77.26   |    75.00    |   85.22   |
| EverMemOS           |    96.67   |   91.84   |  89.72   |    76.04    |   93.05   |
| **MindMemOS-schema** | **97.62**  | **93.26** | 89.01 |  75.00 | **93.64** |

Evaluation config: [`config/mindmemos_eval/memory_evaluation_locomo.example.yaml`](config/mindmemos_eval/memory_evaluation_locomo.example.yaml)

```bash
cp config/mindmemos_eval/memory_evaluation_locomo.example.yaml config/mindmemos_eval/memory_evaluation_locomo.yaml
# fill in API keys, then run:
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo \
  --algorithm schema \
  --manifest-output output/locomo_manifest.jsonl \
  --api-key-output config/mindmemos/api_keys.yaml
```

### Evaluation of Persona Memory

MindMemOS achieves state-of-the-art on PersonaMem through higher-order property modeling and discovery, leading the current SOTA by approximately **2 points** in overall accuracy.

* Benchmark: [PersonaMem](https://arxiv.org/abs/2504.14225), a persona-centric memory benchmark focused on user profiling and preference understanding, evaluating recall, tracking, revisiting, suggestion, recommendation, and generalization of user traits.
* Note: All results are from local runs of open-source code (memory model and answer model: gpt-4.1-mini).

| Method              | Recall Sha. | Recall Men. (Ack. Latest) | Track Evo. | Revisit | Suggest | Recommend | Generalize | Overall          |
| :------------------ | :---------: | :-----------------------: | :--------: | :-----: | :-----: | :-------: | :--------: | :--------------: |
| MemOS               | 74.42% (96/129) | 82.35% (14/17) | 61.87% (86/139) | 77.78% (77/99) | 44.09% (41/93) | 67.27% (37/55) | 84.21% (48/57) | 67.74% (399/589) |
| EverMemOS           | 74.42% (96/129) | 64.71% (11/17) | 64.03% (89/139) | 85.86% (85/99) | 35.48% (33/93) | 65.45% (36/55) | 84.21% (48/57) | 67.57% (398/589) |
| MemU                | 64.34% (83/129) | 64.71% (11/17) | 66.20% (92/139) | 87.88% (87/99) | 31.18% (29/93) | 67.27% (37/55) | 84.21% (48/57) | 65.70% (387/589) |
| **MindMemOS-schema** | 73.64% (95/129) | **82.35%** (14/17) | **67.63%** (94/139) | 85.86% (85/99) | 35.48% (33/93) | **80.00%** (44/55) | 78.95% (45/57) | **69.61% (410/589)** |

### Evaluation of Dreaming

* Benchmark: [MemoryAgentBench](https://arxiv.org/abs/2507.05257), a multi-session agent benchmark measuring Subsequence Exact Match (SubEM) for memory-augmented QA. Compares SubEM performance and memory count changes before and after Dreaming.
* Experiment setting: top-k=50, chunk_size=1024
* Note: MIRIX and mem0 baseline results are from the paper, where chunk_size=4096.

| Pipeline | Memory Model | Judge Model | Answer Model | Single-hop SubEM | Single-hop Gain | Single-hop Memory Count Change | Multi-hop SubEM | Multi-hop Gain | Multi-hop Memory Count Change | Average SubEM | Average Gain | Average Memory Count Change |
|----------|--------------|-------------|--------------|:----------------:|:---------------:|:------------------------------:|:---------------:|:--------------:|:-----------------------------:|:------------:|:------------:|:---------------------------:|
| MIRIX | GPT-4.1-mini | - | GPT-4.1-mini | 20.00% | - | - | 3.00% | - | - | 11.50% | - | - |
| mem0 | GPT-4o-mini | - | GPT-4o-mini | 18.00% | - | - | 2.00% | - | - | 10.00% | - | - |
| Ours (Vanilla) | gpt-4.1-mini | - | gpt-4.1-mini | 83.00% | | - | 10.75% | | - | 46.88% | | - |
| **Ours (Vanilla + Dreaming)** | **gpt-4.1-mini** | **-** | **gpt-4.1-mini** | **88.75%** | **+5.75%** 🟢 | **-27.5%** | **14.00%** | **+3.25%** 🟢 | **-28.3%** | **51.38%** | **+4.50%** 🟢 | **-27.9%** |

## Core Features

- **Portable across agents**: Persist user profiles, preferences, project facts, tool experience, and skill candidates as reusable user assets that can move across OpenClaw, Hermes, Claude Code, OpenHands, and other agent frameworks.
- **Self-evolving memory**: Improve memory quality continuously through schema learning, dreaming, and feedback, so the system can learn frequent memory patterns, consolidate duplicates, and use correction signals to optimize add/search behavior.
- **Memory-Skills loop**: Turn experience memories into skill candidates, then feed skill execution results, failures, and user feedback back into memory so skills can keep evolving.

## Coming Features

- **Skills system**: Route and govern large, redundant skill libraries, evolve skills from real usage, and synthesize new skills from high-frequency user scenarios through offline simulation and refinement.
- **File system memory**: Structure scattered knowledge across local files, documents, project artifacts, and agent outputs into managed knowledge objects or graphs, so agents can use the user's file knowledge more reliably to complete tasks.
- **Agent integrations**: Deeper integrations with coding agents, OpenClaw, Codex-style workflows, and long-running multi-agent systems.

## Quickstart Guide

### 1. Local Deployment

MindMemOS uses `uv` for dependency management and local command execution.

For detailed configuration instructions, see [docs/deploy/instruction.md](docs/deploy/instruction.md).

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```

Configure `config/mindmemos/api_keys.yaml` with an API key and its bound `project_id`. Public memory APIs use bearer authentication:

```text
Authorization: Bearer <api_key>
```

Before starting the stack, edit `config/mindmemos/dev.yaml` and configure at least these model routers:

- `chat_model_router`: LLM generation and extraction.
- `embed_model_router`: semantic embedding generation.
- `rerank_model_router`: reranking for search candidates.

Also make sure `database.qdrant.vector_size` matches the embedding endpoint
dimension.

Start the local stack:

```bash
make dev
```

`make dev` starts the full Docker dependency stack before FastAPI. To start only core dependencies:

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

Docker tier rules:

- `make dev-core` starts Qdrant, Neo4j, Kafka, Kafka UI, and kafka-exporter.
- `make dev` and `make db-observability` start the full Docker dependency stack.
- If `telemetry.enabled=true`, use the full stack so the OTel collector and ClickHouse endpoint exist; Grafana is only for viewing telemetry.

Default local services:

```text
FastAPI:   http://127.0.0.1:8000
API Docs:  http://127.0.0.1:8000/docs
```

Call the API at `http://127.0.0.1:8000` and use the API key configured in
`config/mindmemos/api_keys.yaml`.

Stop the local stack:

```bash
make dev-down
```

### 2. Call FastAPI

Use the API key configured in `config/mindmemos/api_keys.yaml` in the
`Authorization: Bearer <api_key>` header.

Add a memory:

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

Search memories:

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

### 3. Skills Basic Usage

Skills are versioned `SKILL.md` bundles that can be attached to memory add
traces and later evolved from real usage. Register a skill first:

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

The response includes `cloud_skill_id`, `version_id`, and `content_hash`. List
or inspect managed skills:

```bash
curl -X GET http://127.0.0.1:8000/v1/skills \
  -H "Authorization: Bearer <api_key>"

curl -X POST http://127.0.0.1:8000/v1/skills/<cloud_skill_id>/get \
  -H "Authorization: Bearer <api_key>"
```

When writing task traces, pass lightweight `skill_context` references in
`/v1/memory/add`; do not include full skill content there:

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

The Python SDK exposes the same flow through `MindMemOSClient.skills`; registered
local skills can be pushed, pulled, synced, updated, rolled back, and used by
`client.memory.add(...)` automatically when skill detection is enabled.

### 4. Python SDK

The Python SDK is included in this workspace under `src/mindmemos_sdk`.

Configure credentials once:

```bash
uv run mindmemos auth
```

Then reuse the local SDK config:

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

Or pass credentials explicitly:

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

The CLI is shipped with the Python SDK package.

Configure the CLI:

```bash
uv run mindmemos auth
```

Add and search memories:

```bash
uv run mindmemos memory add --content "我喜欢喝冰美式"
uv run mindmemos memory search "咖啡偏好" --top-k 5
```

Manage memories:

```bash
uv run mindmemos memory get --top-k 10
uv run mindmemos memory update <memory_id> --content "我现在更喜欢拿铁"
uv run mindmemos memory delete <memory_id>
uv run mindmemos memory feedback --text "刚才召回的偏好不准确"
uv run mindmemos memory dreaming
```

### 6. OpenClaw Plugin

The OpenClaw plugin is included under `plugins/openclaw-plugin`. It searches
MindMemOS before each prompt and injects relevant memories as
context, then stores completed OpenClaw conversations through the `mindmemos`
CLI.

Authenticate the CLI first:

```bash
uv run mindmemos auth
```

Build, link, and enable the local plugin in OpenClaw:

```bash
npm --prefix plugins/openclaw-plugin install
npm --prefix plugins/openclaw-plugin run build
openclaw plugins install --link /path/to/repo/plugins/openclaw-plugin
openclaw plugins enable mindmemos-memory
```

For full configuration and troubleshooting, see
[plugins/openclaw-plugin/README.md](plugins/openclaw-plugin/README.md).

## License

MIT License.
