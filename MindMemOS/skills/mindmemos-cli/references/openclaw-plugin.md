# OpenClaw integration

The OpenClaw memory plugin wires MindMemOS into OpenClaw automatically: it
`search`es memories before every prompt and injects the hits as context, and
`add`s finished conversations at the end of a turn. It is a thin hook layer over
the `mindmemos` CLI.

**Prerequisite:** the `mindmemos` CLI must be installed and authenticated first
(see the main SKILL.md → "Install the CLI"). Installing the plugin alone does
nothing — it shells out to the CLI for every operation.

## Install the plugin

Build the plugin from this workspace and link its package root through OpenClaw:

```bash
npm --prefix plugins/openclaw-plugin install
npm --prefix plugins/openclaw-plugin run build
openclaw plugins install --link /path/to/repo/plugins/openclaw-plugin
openclaw plugins enable mindmemos-memory
```

`mindmemos-memory` is the plugin id from its manifest.

### Grant conversation access (required, or nothing gets stored)

The plugin reads conversation messages in order to persist them, so it needs
conversation access on its `agent_end` hook. The manifest declares
`hooks.allowConversationAccess=true`, but for a **non-bundled** linked plugin
that is not shipped inside OpenClaw, that manifest
field is only a *request* — OpenClaw does not honor it. You must also grant it
explicitly at the entry level in config:

```bash
openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
openclaw gateway restart
```

This writes `plugins.entries["mindmemos-memory"].hooks.allowConversationAccess
= true` into `~/.openclaw/openclaw.json`. Without it, recall still works (the
pre-prompt `search` hook does not need conversation access), so the plugin looks
healthy — but every `agent_end` store is silently blocked. See Troubleshooting
below for how to confirm.

## Configure

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

- `cli` — command used to invoke the CLI. Defaults to `mindmemos`. If OpenClaw does not run with the CLI on its PATH, set this to an absolute path or a wrapper command (e.g. `uv run mindmemos` inside the repo).
- `topK` — number of memories injected per turn.
- `addMode` — `sync` blocks until extraction finishes; `async` (default) enqueues and returns immediately. Note: in `async` mode the plugin only sees CLI-level failures — a server-side add failure that happens after the CLI exits 0 is not logged by the plugin.
- `userId` / `appId` / `sessionId` — override scoping; otherwise the plugin derives them.
- `minQueryLength` — skip recall for very short prompts.
- `maxConversationMessages` — cap on how many trailing messages are persisted per turn.

## Subagent behavior

In subagent sessions (session key contains `:subagent:`), recalled memories are
injected as **background context only**, with a preamble telling the subagent
not to assume it is the user. This avoids a subagent adopting the main user's
identity.

## Troubleshooting

- CLI errors surface in the OpenClaw gateway log as `[mindmemos] memory search failed: ...` / `[mindmemos] memory add failed: ...` warnings, including the CLI's stderr / exit code.
- If recall never fires, check `minQueryLength` and that `cli` resolves to a working `mindmemos` command (`mindmemos config show` from the same environment OpenClaw runs in).

### Recall works but nothing is ever stored (`agent_end blocked`)

Symptom: the gateway log shows recall running every turn
(`[mindmemos] ... hit N memories, injected ... chars`) but you never see a
`[mindmemos] stored N message(s) from last turn` line, and on plugin load the
gateway prints:

```
[gateway] [plugins] typed hook "agent_end" blocked because non-bundled plugins
must set plugins.entries.mindmemos-memory.hooks.allowConversationAccess=true
```

Cause: OpenClaw's security policy ignores the manifest's
`hooks.allowConversationAccess` for non-bundled plugins; only the entry-level
config field is honored. The pre-prompt `search` hook does not need conversation
access, which is why recall keeps working and the failure is easy to miss.

Fix: grant access in config and restart (see "Grant conversation access" above):

```bash
openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
openclaw gateway restart
```

Confirm the fix:

- On restart, `[mindmemos] plugin loaded` appears **without** the `agent_end
  blocked` warning.
- Recall still fires as before.
- After the **next** turn completes, the log shows `[mindmemos] stored N
  message(s) from last turn`. The store only runs at `agent_end`, so it will not
  appear until a turn finishes — checking mid-turn always looks empty.

### `mindmemos` not found (PATH mismatch)

The plugin spawns the CLI using the PATH of the **OpenClaw process**, which is
not always the PATH of your interactive shell. A GUI-launched OpenClaw (Desktop
app, or one started by a launch agent rather than from a terminal) typically
inherits a minimal PATH that does **not** include `~/.local/bin`,
`~/.cargo/bin`, Homebrew dirs, or a project virtualenv — so a `mindmemos` that
works in your terminal can still be invisible to the plugin. Symptom: every turn
logs `[mindmemos] ... failed: ... ENOENT` or `command not found`, even though
`mindmemos` runs fine for you.

Diagnose:

```bash
# 1. Does it work in your shell at all, and where does it live?
which mindmemos          # e.g. /Users/you/.local/bin/mindmemos

# 2. Reproduce the plugin's view: run the SAME way OpenClaw was launched.
#    If OpenClaw is a GUI app, check the PATH it actually sees — the env that
#    spawned it, not your shell. From a terminal that DID start it:
echo "$PATH"
# 3. Confirm the gateway log shows ENOENT / not-found rather than an auth or
#    network error (those mean the CLI was found and the problem is elsewhere).
openclaw plugins doctor
```

Fix (pick one):

- **Point `cli` at an absolute path** (most reliable, no PATH dependency):
  set the plugin config `"cli": "/Users/you/.local/bin/mindmemos"` (use the path
  from `which mindmemos`).
- **Use a wrapper that resolves the CLI itself**, e.g. `"cli": "uv run mindmemos"`
  when OpenClaw's working directory is this repo.
- **Use the workspace virtualenv executable directly**, such as
  `/path/to/repo/.venv/bin/mindmemos`.

After changing `cli`, re-run a turn and confirm the gateway log no longer shows
the spawn error.
