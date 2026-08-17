Record the just-finished metric-backed experiment as one compact stage.
This is a bookkeeping turn only: do not edit modeling or training files, do not run training, and do not inspect extra files.
Append exactly one new stage entry to `{ledger_filename}` headed `### {stage_id}`. Preserve all existing entries unchanged.
Required entry format:
### {stage_id}
metric: <float>
lower_is_better: <true|false>
run_time_sec: <seconds|unknown>
metric_type: <holdout|cv|post_fulltrain_replay|training_loss|unknown>
metric_note: <one concise line explaining the validation metric semantics>
metric_validity: <high|medium|low>
BRIEF: <one concise line describing model/features/training choices>
WHY: <one concise line explaining why this stage is worth preserving, including any route lesson or avoid-repeat evidence>
FILES: <code=core.py,helper.py weights=model.ckpt or none>

Use `WHY` for the route judgment too. Do not add a separate `route_evidence` field. Do not summarize process logs or repeat metrics already in structured fields. Use `FILES` only for core workspace `.py` files and ML model weights needed to continue the method; exclude outputs, submissions, result JSON, logs, temp files, and caches.

Use the metric event below as the source of truth for metric fields. For `run_time_sec`, copy `run_time_sec` if present, otherwise use `wall_sec` or `duration_sec`; write `unknown` only if unavailable. For `metric_type`, copy `val_score_type`; for `metric_note`, summarize `selection_note` or state that the metric is eligible holdout/CV evidence. For `metric_validity`, write `high` only for comparable held-out/OOF/CV metrics, `medium` for useful proxy/replay evidence, and `low` for leakage, schema, invalid-submission, or train+validation re-evaluation risk. Preserve existing stages.
Metric event JSON: {metric_json}

Current `{ledger_filename}`:
{prior_block}

