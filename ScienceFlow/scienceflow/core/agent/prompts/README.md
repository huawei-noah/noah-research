# ScienceAgent Prompt Templates

Base system prompt (L0) for ScienceAgent. `default_system.md` contains workspace
rules, behaviour guidelines, and reasoning discipline; tool policy is injected
via `{policy}` placeholder from either `tool_policy_single.md` or
`tool_policy_parallel.md`.

These templates form the shared base for both REPL and LNR usage. LNR layers
(L0 contract/rules, L1 pinned, L2 dynamic) are appended by LNR-specific hooks.
