# LNR Resume

This package keeps resume logic out of the main LNR solver. The design goal is
to continue from persisted agent memory instead of injecting a new resume summary
into the main-agent prompt.

## Resume levels

1. Task-level resume: the parallel runner reuses the same workspace and adjusts
   remaining budget from `logs/state.json`.
2. Agent-memory resume: LNR reloads the existing `ScienceAgent` memory and decides
   whether the next action is an LLM round or a pending tool execution.
3. Tool-cursor resume: when the memory tail is an assistant tool call without a
   matching tool result, LNR executes that tool call directly and writes the tool
   result back to memory before the next LLM round.

## No prompt pollution

Resume events are audit records only. They are not appended as user messages. The
main agent continues from the persisted memory transcript.

## Current support

The first implementation supports a single pending tool call at the memory tail,
which covers the common interruption point after the LLM decided to run a command
such as `python train.py` but before the tool result was recorded. Multiple
pending tool calls are detected and audited but not replayed automatically yet.
