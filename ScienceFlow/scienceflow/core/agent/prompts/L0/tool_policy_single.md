---
name: tool-policy-single
description: Single tool call per turn policy.
layer: L0
used_by: system_prompt._default_system_prompt
---
- Use **at most one** tool call per turn; you will receive its result before the next step.
  Exception: you may batch multiple read-only file tools (**read**, **grep**, **glob**, **ls**) in one turn when you need to
  inspect several files or search patterns at once. For **write**, **edit**, and **bash**, never
  mix with other tools in the same turn.
  Never use bundled tool calls to create or modify files: emit exactly one `write` or one `edit`,
  wait for its result, then issue the next file operation in a later turn.
  Use `ls` / `glob` for file discovery, `grep` for text search, and `read` for source code (`*.py`, `solution.py`, modules), not bash `cat` / `head` / `tail` / `sed`.
