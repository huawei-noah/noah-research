# OpenClaw MindMemOS Memory Plugin

This plugin connects OpenClaw conversations to MindMemOS through the local
`mindmemos` CLI.

- On every user turn, it searches memories and injects the hits as prompt context.
- At the end of an agent turn, it writes that turn's messages to MindMemOS with
  `memory add` (the last user message and everything that follows it).

## Captured messages

OpenClaw's transcript carries several message types. The plugin normalizes and
persists the conversational ones:

- `user` and `assistant` messages — including assistant tool calls, which are
  serialized inline as `[tool_call] <name>(<json args>)` and kept on the
  assistant message.
- `toolResult` messages — mapped to the generic `tool` role.

Other OpenClaw harness message types are intentionally dropped, because they are
internal artifacts rather than conversation content: `bashExecution`, `custom`,
`branchSummary`, and `compactionSummary`. The system prompt is not a transcript
message in OpenClaw, so it is never persisted either.

## Requirements

Install and authenticate the `mindmemos` CLI once:

```bash
mindmemos auth
```

Build the plugin from this workspace and link this package root:

```bash
npm install
npm run build
openclaw plugins install --link /path/to/repo/plugins/openclaw-plugin
openclaw plugins enable mindmemos-memory
```

Installing with `--link` points the install at this source directory, so a later
`npm run build` takes effect without reinstalling. After install/enable, restart
the gateway to load the plugin:

```bash
openclaw gateway restart
```

Do not install `src/` or `src/index.ts` directly. OpenClaw expects the native
plugin manifest at the plugin root (`openclaw.plugin.json`). `openclaw plugins
validate` is only for simple `defineToolPlugin` tool plugins, so it is not the
right validation command for this hook-based memory plugin.

### Grant conversation access

The plugin persists conversation messages, so it needs conversation access. The
manifest declares `hooks.allowConversationAccess=true`, but for a non-bundled
(linked/installed) plugin that is only a request — you must also grant it in
config, otherwise the gateway blocks the `agent_end` write hook:

```bash
openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
openclaw gateway restart
```

If the gateway log shows `typed hook "agent_end" blocked because non-bundled
plugins must set ... allowConversationAccess=true`, this grant is missing.

## Configuration

Optional plugin config:

```json
{
  "enabled": true,
  "cli": "mindmemos",
  "topK": 5,
  "addMode": "async",
  "userId": "optional-user-override",
  "appId": "openclaw",
  "sessionId": "optional-session-override",
  "minQueryLength": 2,
  "maxConversationMessages": 80
}
```

The `cli` value is spawned by the gateway, which (under launchd/systemd) does
not inherit your shell `PATH`. The default `"mindmemos"` therefore fails with
`spawn mindmemos ENOENT` unless `mindmemos` is on the service PATH. Prefer an
absolute path to the executable, for example:

```bash
openclaw config set plugins.entries.mindmemos-memory.config.cli \
  /path/to/repo/.venv/bin/mindmemos
```

The `.venv/bin/mindmemos` shim has an absolute-path shebang, so it runs without
activating the venv. Alternatively set `cli` to a wrapper such as
`uv run mindmemos`, but only if the gateway can resolve `uv` and the repo
working directory. If memory add/search silently does nothing, check the gateway
log for `ENOENT` — that means the `cli` path is wrong.

## Subagent Behavior

In subagent sessions (session key contains `:subagent:`), recalled memories are
injected as background context only, with a preamble telling the subagent not to
assume it is the user. This avoids a subagent adopting the main user's identity.

## Skill Recognition

When the last turn contains OpenClaw tool-call text for `read`, `write`, or
`edit` on a `SKILL.md` path, the plugin extracts the skill content from the same
messages it sends to `memory add`, computes the SDK-compatible content hash, and
passes it via `--skill-context-json`. This lets MindMemOS bind memories to the
skill version that was loaded or modified without reading OpenClaw's private
trace format.

## License

MIT License.
