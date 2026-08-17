---
name: tool-policy-parallel-bash-write
description: Parallel tool calls per turn policy for REPL bash file-write mode.
layer: L0
used_by: system_prompt._default_system_prompt(bash_file_write_mode=True)
---
- **Use multiple tool calls per turn** to speed up data understanding and exploration.
  Independent read-only bash commands and optional read-only file tools (`read`, `grep`, `glob`, `ls`) run concurrently.
  * Prefer compact bash commands for inspection/search/data summaries. Use `read` only for exact line-numbered source context or stable edit anchors.
  * Do not spend multiple rounds paging through `read`; switch to a compact bash/search summary or make the next targeted change.
  * **NEVER** chain independent commands with `&&` - emit separate bash tool calls instead.
    Bad: one bash `head a.csv && wc -l b.csv`  Good: two bash tools, one per command.
  * Avoid dumping whole files through bash. Bounded `sed -n`, `rg -n`, `wc`, `stat`, and short Python summaries are acceptable.
  * If a read-only bash command may print many lines, make it produce a compact summary or write full output to a workspace log file and print the log path plus a small tail.
  * If a tool result says output was truncated or summarized, ask a narrower follow-up question with `grep`, paged `read`, or a more selective command before relying on missing details.
  * For verbose validation/training, do not use `command | tail` as the only output record. Write the full output to workspace scratch such as `tmp/...` first, then print metrics and a compact tail.
  * Any side-effect bash command that modifies files, installs packages, runs git, or launches validation/training must be the **sole** call in a turn.
  * **Never use parallel or bundled tool calls to create or modify files.** Emit exactly one side-effect bash command, wait for its result, then issue the next file operation in a later turn.
  * When writing a script with a heredoc, do not execute it in the same bash command. Write/update first, wait for the result, then run the script as a separate foreground bash command.
  * On an existing file, prefer targeted shell-level changes over full replacement when only a small region changes.
