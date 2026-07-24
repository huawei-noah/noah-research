---
name: mindmemos-cli
description: Give an AI agent persistent, cross-session long-term memory through MindMemOS. Covers installing and authenticating the mindmemos CLI, the full command interface (add / search / get / update / delete / feedback / dreaming) with parameters and examples, guidance on which capability to use when, plus a Python SDK example. To wire memory into a specific agent host (OpenClaw, Codex, Claude, etc.), see references/.
---

# MindMemOS CLI

MindMemOS is a long-term memory layer for AI agents. The `mindmemos` CLI is the
integration surface: every memory operation is a subcommand that prints either a
human-readable line or, with `--json`, stable machine-readable output. Any agent
or script can drive memory by shelling out to it.

To connect memory to a specific agent host (e.g. an editor or assistant that
supports plugins), the host calls this same CLI. Host-specific install guides
live under `references/` — see [Host integrations](#host-integrations).

---

## Install the CLI from this workspace

The CLI is included in this workspace and exposes a `mindmemos` executable.

```bash
uv sync
```

Authenticate once. This writes a local config (API key, default user id, base URL):

```bash
uv run mindmemos auth
# non-interactive:
uv run mindmemos auth --api-key sk-... --user-id alice --base-url http://127.0.0.1:8000
```

Verify:

```bash
uv run mindmemos config show          # masked key, base_url, user_id
uv run mindmemos memory search "test" # confirms connectivity
```

---

## CLI interface

General shape: `mindmemos <group> <command> [args] [options]`.

- Memory commands do not accept a caller-provided request ID. The server generates
  `request_id` and includes it in command responses for tracing.
- `search` / `add` support `--json` for stable machine-readable output (what scripts and host integrations parse).
- Exit codes: `0` = success, `1` = API/config error, `2` = bad arguments. On non-zero exit the error text (including server stderr) is printed to stdout/stderr.

Identity & scoping options (where accepted): `--user-id` (the human the memory
belongs to), `--app-id`, `--agent-id`, `--session-id`. Project isolation is
derived from the API key, not from these flags.

### Typical flow

1. `mindmemos auth` once.
2. During a session: `memory search` to recall, `memory add` to store turns.
3. Maintenance / background: `memory get` to inspect, `memory update` / `memory delete` to correct, `memory feedback` and `memory dreaming` to let the system consolidate.

### `memory add` — store new memory

Extracts durable facts from messages and persists them (with dedup/merge against existing memory).

| Option | Meaning |
|---|---|
| `--content TEXT` | single message body (paired with `--role`) |
| `--role {user,assistant,system,tool}` | role for `--content` (default `user`) |
| `--messages-json '[...]'` | JSON array of messages; overrides `--content` |
| `--messages-json-file PATH` | read the JSON array from a file (`-` = stdin) |
| `--user-id`, `--app-id`, `--agent-id`, `--session-id` | scoping |
| `--metadata-json '{...}'` | business metadata object |
| `--skill-context-json '[...]'` | explicit skill trace context |
| `--async` | enqueue and return immediately (no extracted memories in response) |
| `--json` | machine-readable output |

```bash
# single line
mindmemos memory add --content "I'm allergic to peanuts" --user-id alice

# a conversation turn
mindmemos memory add --messages-json \
  '[{"role":"user","content":"book me a window seat next time"},
    {"role":"assistant","content":"Noted, window seats going forward."}]' \
  --session-id sess-42 --json

# fire-and-forget
mindmemos memory add --content "prefers dark mode" --async
```

### `memory search` — recall by relevance

| Option | Meaning |
|---|---|
| `query` (positional) | search text |
| `--top-k N` | results to return (default 10) |
| `--search-strategy {fast,agentic}` | `fast` = vector recall; `agentic` = multi-step reasoning over memory |
| `--rerank` | rerank candidates for precision |
| `--score-threshold N` | minimum rerank relevance score (0–1); only effective with `--rerank` |
| `--filter '{...}'` | structured filter DSL, JSON object (e.g. `{"memory_type":"semantic"}`) |
| `--user-id`, `--app-id`, `--agent-id`, `--session-id` | scoping |
| `--json` | machine-readable output |

```bash
mindmemos memory search "what are the user's dietary restrictions?" --top-k 5 --user-id alice
mindmemos memory search "travel prefs" --rerank --search-strategy agentic --json
mindmemos memory search "notes" --filter '{"memory_type":"semantic"}'
```

### `memory get` — list / filter (no query)

Returns memories in the current project, optionally filtered. Carries **no**
actor identity — project scope comes from the API key.

```bash
mindmemos memory get --filter '{"app_id":"openclaw"}' --top-k 20
```

### `memory update` / `memory delete` — correct by id

```bash
mindmemos memory update mem_123 --content "allergic to peanuts and shellfish"
mindmemos memory delete mem_123 --yes
```

### `memory feedback` — reinforce / correct memory quality

| Option | Meaning |
|---|---|
| `--text TEXT` | explicit feedback text; omit to let the server analyze recent adds |
| `--user-id`, `--app-id`, `--agent-id`, `--session-id` | scoping |

```bash
mindmemos memory feedback --text "the lunch recommendation was wrong; user dislikes spicy food"
mindmemos memory feedback   # omit --text: server analyzes recent adds
```

### `memory dreaming` — consolidation pass

| Option | Meaning |
|---|---|
| `--sync` | run synchronously |
| `--async` | enqueue asynchronously (default) |
| `--user-id`, `--app-id`, `--agent-id`, `--session-id` | scoping |

```bash
mindmemos memory dreaming
mindmemos memory dreaming --sync --app-id openclaw
```

### Other groups

- `mindmemos auth` / `config show [--show-secret]` / `config reset [-y]` — credentials & local settings.
- `mindmemos skill <register|list|show|pull|push|update|rollback|history|diff|unregister>` — SDK-managed skills. Use `register <skill_dir_or_SKILL.md> --alias <alias>` to save a local alias, then use that alias anywhere a skill id is accepted. Use `push <skill>` after editing local `SKILL.md` to upload a new version. Use `update <skill|--all> [--yes]` to checkout published heads, `rollback <skill> --to <version_id> [--yes]` to restore a cached/downloaded version after reviewing the replacement plan, and `diff <skill> [--from <version_id>] --to <version_id>` for a read-only unified diff.
- `mindmemos memory add ... --skill-context-json '[...]'` — optional explicit skill trace context. When omitted, the SDK has a best-effort fallback for OpenClaw-style `SKILL.md` tool-call text in the add messages; host integrations such as the OpenClaw plugin may still provide their own detection and pass this flag explicitly.
- `mindmemos doctor` — config/connectivity check.

---

## Capabilities — when to use what

MindMemOS is a memory **lifecycle**, not just a key-value store. Pick the
operation by intent:

| Intent | Use | Notes |
|---|---|---|
| "Remember this" — a new fact, preference, or conversation turn surfaced | **`add`** | Server extracts durable facts and dedups/merges against existing memory. Prefer passing real conversation messages over hand-written summaries. |
| "What do I already know about X?" — pull context before answering | **`search`** | Relevance-ranked. `fast` for latency-sensitive recall; `agentic` when the answer requires reasoning across several memories; add `--rerank` when precision matters more than speed. |
| "Show me everything in this project / a slice of it" | **`get`** | Filter/enumerate without a query; for inspection, audits, dashboards. |
| "That stored memory is wrong / stale" | **`update`** (fix one) or **`delete`** (remove one) | Targeted by `memory_id`, which you get from `search`/`get`. |
| "Tell the system how it did" — the recalled/produced memory was right or wrong | **`feedback`** | Signals to reinforce or correct memory quality; with no `--text`, the server analyzes recent adds itself. Use after an interaction whose outcome reveals memory quality. |
| "Consolidate in the background" — compress, link, reorganize accumulated memory | **`dreaming`** | An offline maintenance pass with no inputs. Run periodically (e.g. scheduled), not per-turn. |

Rules of thumb:

- **`add` + `search` are the hot path** — almost every agent turn does one or both.
- **`feedback` and `dreaming` are the slow path** — they improve memory *quality* over time. `feedback` is event-driven (an outcome happened); `dreaming` is schedule-driven (periodic consolidation), not for a hot request path.
- **`update` / `delete` / `get` are manual curation** — fixing mistakes and inspecting state, usually by a human or an admin tool, not in normal conversation flow.
- Always scope writes and reads with a stable `--user-id` (and `--session-id` where it matters) so memories don't leak across users.

---

## Calling from Python

When memory operations live inside a Python agent/app rather than a shell call,
use the SDK shipped in the same `mindmemos` package (same API, same `mindmemos
auth` config). See [references/python-sdk.md](references/python-sdk.md) for the
full sync + async example. Minimal sync usage:

```python
from mindmemos_sdk import MindMemOSClient, DialogueMessage

with MindMemOSClient(user_id="alice") as client:   # reads `mindmemos auth` config
    client.memory.add(messages=[DialogueMessage(role="user", content="allergic to peanuts")])
    hits = client.memory.search("dietary restrictions", top_k=5)
    for hit in hits.memories:
        print(hit.id, hit.memory)
```

---

## Host integrations

To wire MindMemOS into an agent host so memory is recalled and stored
automatically (rather than calling the CLI by hand), follow the host-specific
guide. All hosts depend on the CLI installed and authenticated above.

- **OpenClaw** — [references/openclaw-plugin.md](references/openclaw-plugin.md)
- _Codex_ — planned
- _Claude_ — planned
