Choose the next research direction for one ML search trajectory. A estra is a stage-boundary research judgment, not just a metric selector.
Decide two axes:
- `startpoint`: `current_workspace` or `previous_stage`.
- `intent`: `continue` or `redirect`.

Startpoint semantics:
- `current_workspace` means the next stage continues from the current files. This preserves recent implementation context and is preferred when the current workspace remains a good base for the next experiment.
- `previous_stage` means the next stage starts from a selected historical stage snapshot. This is useful when an earlier stage is a cleaner or more focused experimental base. The target stage does not have to be the best metric stage if it is the better workspace for testing the next hypothesis.

First summarize the exploration briefly, then state the bottleneck and evidence, then choose the action.

Decision meanings:
- `current_workspace` + `continue`: the current route and current execution tactic are still reasonable.
- `current_workspace` + `redirect`: keep the current workspace, but recent work is shallow, repetitive, or not addressing the bottleneck; redirect the next stage around the stated bottleneck.
- `previous_stage` + `continue`: restore one listed historical completed stage because it is a better restart point and continue that stage's route.
- `previous_stage` + `redirect`: restore one listed historical completed stage, but start the next stage with a redirect around the stated bottleneck.

Metric is evidence, not the only selection rule. A switch target may be preferable even without the best metric when it has stronger route potential, cleaner validation, better generalization evidence, useful complementarity to other branches, or a more recoverable code/checkpoint state.
Do not switch solely because a stage has the best metric if its route is likely saturated, leaky, fragile, or less worth further exploration than another listed stage.
If peer route evidence is present, explicitly account for it: adapt the reliable peer idea, restore a local stage to test it, or state why the current route should remain independent.
If backtrack reflection is present, compare current continue, current redirect, and previous-stage redirect before deciding. Returning to a historical stage is a normal research action, not a failure.
Do not classify the bottleneck with a label. State the actual bottleneck in a concrete short sentence.

Length limits:
- exploration_summary <= 25 words
- bottleneck <= 25 words
- evidence <= 25 words
- missing_evidence <= 20 words or null
- decision_reason <= 25 words
- redirect_focus <= 20 words or null
- all diagnostic text combined <= 120 words

Current/latest stage: {latest_stage}
Switch-stage candidates: {switch_candidates}
For `previous_stage`, target_stage must be one listed Sxx stage. For `current_workspace`, omit target_stage or use the current/latest stage.
Do not call tools, write files, use markdown, or emit DSML/tool markup.
Return exactly one JSON object only with this shape:
{{"startpoint":"current_workspace|previous_stage","intent":"continue|redirect","exploration_summary":"...","bottleneck":"...","evidence":"...","missing_evidence":null,"is_route_flaw":false,"is_execution_flaw":true,"target_stage":null,"decision_reason":"...","redirect_focus":null}}

Current `{ledger_filename}`:
{ledger}{trailing_blocks}
