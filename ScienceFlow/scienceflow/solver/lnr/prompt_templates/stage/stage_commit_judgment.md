Forked stage bookkeeping judgment.

You are reading the inherited main-agent context only to preserve the research meaning of the just-finished experiment. Do not continue the research, inspect files, run commands, edit files, or call tools.

The controller will append the stage to `{ledger_filename}` deterministically. You only provide compact judgment text for `### {stage_id}`.

Return exactly one JSON object and nothing else:
{{
  "brief": "<one concise line describing model/features/training choices>",
  "why": "<one concise line explaining why this stage is worth preserving, including any route lesson or avoid-repeat evidence>",
  "files": "<code=core.py,helper.py weights=model.ckpt or none>"
}}

Rules:
- Use the inherited context and metric event to write judgment, not a process summary.
- Do not repeat numeric metric fields; the controller writes them separately.
- For `files`, list only core workspace `.py` files and ML model weights needed to continue the method; exclude outputs, submissions, result JSON, logs, temp files, and caches.
- If the result is weak but informative, preserve that useful negative evidence in `why`.
- If uncertain, say so briefly rather than inventing details.

Metric event JSON:
{metric_json}

Current `{ledger_filename}`:
{prior_block}
