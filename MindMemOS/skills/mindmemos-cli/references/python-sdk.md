# Python SDK usage

Besides the CLI, MindMemOS ships a Python SDK in the same `mindmemos-sdk` package.
Use it when memory operations live inside a Python agent/app rather than a shell
call. The SDK and CLI hit the same API and share the same local config written by
`mindmemos auth`.

## Setup from this workspace

```bash
uv sync
```

Run CLI commands from the workspace with `uv run mindmemos`.

Credentials resolve in this order: explicit constructor args → the local config
from `mindmemos auth` → empty (calls then raise `AuthRequiredError`).

## Synchronous client

`MindMemOSClient` is the root entry; `client.memory` exposes the memory
operations. It is a context manager that releases the underlying HTTP transport
on exit.

```python
from mindmemos_sdk import MindMemOSClient, DialogueMessage
from mindmemos_sdk import MindMemOSSDKError

# No args -> reads base_url / api_key / user_id from `mindmemos auth` config.
# Or override explicitly:
with MindMemOSClient(
    # base_url="http://127.0.0.1:8000",
    # api_key="sk-...",
    user_id="alice",
    app_id="my-agent",
    session_id="sess-42",
) as client:
    # 1) Store a conversation turn
    add_result = client.memory.add(
        messages=[
            DialogueMessage(role="user", content="I'm allergic to peanuts"),
            DialogueMessage(role="assistant", content="Noted — I'll avoid peanuts."),
        ],
        # mode="async",            # enqueue and return; memories will be empty
        # metadata={"channel": "chat"},
    )
    print(add_result.code, add_result.request_id)
    for item in add_result.memories:
        print(item.operation, item.memory_id, item.content)

    # 2) Recall before answering
    search_result = client.memory.search(
        "what are the user's dietary restrictions?",
        top_k=5,
        search_strategy="fast",     # or "agentic"
        rerank=True,
        score_threshold=0.5,        # filter out low-relevance rerank results
    )
    for hit in search_result.memories:
        print(hit.id, hit.memory, hit.last_update_at)

    # 3) Inspect / curate
    got = client.memory.get(filters={"app_id": "my-agent"}, top_k=20)
    client.memory.update("mem_123", "allergic to peanuts and shellfish")
    client.memory.delete("mem_123")

    # 4) Quality lifecycle
    client.memory.feedback(feedback="lunch rec was wrong; user dislikes spicy food")
    client.memory.dreaming()        # background consolidation, no inputs
```

`messages` accepts typed objects (`DialogueMessage`, `TextMessage`, `UrlMessage`,
`FileMessage`) or equivalent dicts. Per-call kwargs (`user_id`, `app_id`,
`agent_id`, `session_id`) override the client defaults. Request IDs are generated
by the server and returned in SDK results and API errors; callers cannot provide
them.

Useful memory kwargs:

- `add(..., mode="sync"|"async", metadata={...}, skill_context=[...], score=0.8, task_id="...")`
- `search(..., search_strategy="fast"|"agentic", rerank=True, score_threshold=0.5, filters={...})`
- `feedback(feedback="...", mode="sync"|"async", messages=[...], recalled_memories=[...])`
- `dreaming(mode="async"|"sync")`

Error handling — all API/SDK failures raise `MindMemOSSDKError` (subclasses:
`AuthRequiredError`, `TransportError`, `ApiError`); `ApiError` carries
`request_id`:

```python
try:
    client.memory.search("notes", user_id="alice")
except MindMemOSSDKError as exc:
    print("memory call failed:", exc, getattr(exc, "request_id", None))
```

## Asynchronous client

There is no async root client; build an `AsyncHttpTransport` and pass it to
`AsyncMemoryClient`. The transport is an async context manager.

```python
import asyncio
from mindmemos_sdk import AsyncMemoryClient
from mindmemos_sdk.transport import AsyncHttpTransport

async def main():
    async with AsyncHttpTransport(
        base_url="http://127.0.0.1:8000",
        api_key="sk-...",
    ) as transport:
        memory = AsyncMemoryClient(transport, default_user_id="alice", default_app_id="my-agent")

        await memory.add(messages=[{"role": "user", "content": "prefers window seats"}])
        result = await memory.search("seating preference", top_k=3)
        for hit in result.memories:
            print(hit.id, hit.memory)
        await memory.get(filters={"app_id": "my-agent"}, top_k=20)
        await memory.update("mem_123", "prefers aisle seats")
        await memory.delete("mem_123")
        await memory.feedback(feedback="seat preference was recalled correctly")
        await memory.dreaming(mode="async")

asyncio.run(main())
```

## Result shapes

- `add` → `AddResult(code, request_id, memories=[{operation, memory_id, content}])`. In `async` mode `memories` is empty and `code` is `"queued"`.
- `search` / `get` → `SearchResult` / `GetResult` with `memories=[{id, memory, last_update_at, ...}]`.
- `update` / `delete` / `feedback` / `dreaming` → `StatusResult(code, message, request_id)`.

## Skill version commands

SDK-managed skills are controlled through the CLI:

```bash
mindmemos skill register ./skills/demo/SKILL.md --alias demo
mindmemos skill pull <skill_id_or_alias>
mindmemos skill update <skill_id_or_alias> --yes
mindmemos skill rollback <skill_id_or_alias> --to <version_id> --yes
mindmemos skill diff <skill_id_or_alias> --from <version_id> --to <version_id>
```

`register` accepts either a skill directory or its `SKILL.md` path; the SDK stores the parent directory as the managed path. `register --alias` stores a local alias in the SDK registry. All single-skill commands accept either the generated `skill_id` or that alias. `update` checks the cloud published head and applies the same checkout/backup flow as rollback. `rollback` downloads missing target content into the local cache, prints a replacement plan with file, hash, and backup path details, then applies only after confirmation unless `--yes` is set. `diff` is read-only: it compares cached or downloaded version content and never changes the working skill directory.

`memory add` also accepts explicit skill traces:

```bash
mindmemos memory add --messages-json-file turn.json \
  --skill-context-json '[{"name":"demo","content_hash":"...","base_version_id":"v1","usage":"injected"}]'
```

When `--skill-context-json` is omitted, the sync SDK/CLI has a best-effort fallback for OpenClaw-style `[tool_call] read|write|edit({"path":".../SKILL.md"})` messages in the same add payload: it computes the canonical bundle hash, ensures registered local skills when possible, and sends `skill_context` with the add request. Host integrations such as the OpenClaw plugin can still run their own skill detection and pass `--skill-context-json` explicitly.
