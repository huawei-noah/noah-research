# mindmemos-sdk

Python SDK and CLI for MindMemOS, a long-term memory system for AI agents and applications.

## Install from this workspace

```bash
uv sync
```

Run the `mindmemos` command with `uv run mindmemos`.

## Configure

```bash
mindmemos auth
```

You can also pass `base_url`, `api_key`, and `user_id` directly when creating a client.

## Python SDK

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

with MindMemOSClient(user_id="alice", app_id="my-agent") as client:
    client.memory.add(
        messages=[
            DialogueMessage(role="user", content="I prefer iced Americano."),
        ],
    )

    result = client.memory.search("What coffee does the user prefer?", top_k=5)
    for memory in result.memories:
        print(memory.memory)
```

## CLI

```bash
mindmemos memory add --content "I prefer iced Americano" --user-id alice
mindmemos memory search "coffee preference" --top-k 5 --user-id alice
```
