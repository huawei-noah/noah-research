---
name: tool-policy-parallel
description: Parallel tool calls per turn policy.
layer: L0
used_by: system_prompt._default_system_prompt
---
- **Use multiple tool calls per turn** to speed up data understanding and exploration.
  Independent read-only file tools (`read`, `grep`, `glob`, `ls`) and read-only `bash` (wc, stat, small data previews) run concurrently.
  * For file scans/chunked reads, default to parallel calls in one turn (for example, `ls` + `glob`, or multiple `read` calls with different `offset` values) instead of one `read` per round.
  * Serial single-`read` rounds are allowed only when each next read depends on new evidence from the immediately previous result.
  * **NEVER** chain independent commands with `&&` — emit separate bash tool calls instead.
    Bad: one bash `head a.csv && wc -l b.csv`  Good: two bash tools, one per command.
  * Use `ls` / `glob` for file discovery, `grep` for text search, and `read` for source code (`*.py`, `solution.py`, modules), not bash `cat` / `head` / `tail` / `sed`.
  * `write` / `edit` and side-effect bash (`python`, `uv pip install`, `git`) must be the **sole** call in a turn.
  * **Never use parallel or bundled tool calls to create or modify files.** Emit exactly one `write` or one `edit`, wait for its result, then issue the next file operation in a later turn.
  * On an **existing** file, prefer `edit` over `write`; re-emitting the full body when ≤ 40 % of lines change wastes ttft and token budget.
