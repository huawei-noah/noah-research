You are solving one ML task in a continuous REPL workspace.
Use only relative workspace paths such as `dataset/`, source files, artifacts, an entry script, and `submission.csv`.
Inspect the data, build or improve the workspace pipeline, and run validation; `solution.py` is acceptable but not required.
Choose experiment scale from observed or reasonably estimated cost. A simple full run is appropriate when it or its first comparable held-out/CV metric should finish within about 15 minutes. For an unvalidated route expected to take longer, first run a bounded pilot that preserves the same metric and pipeline structure while reducing samples, epochs, folds, resolution, estimators, or model size. Scale to full data only after that route produces a comparable metric; reuse the pilot code and caches to avoid train/test drift. Make unavoidable long full-scale work resumable and identify its next measurable metric or reusable artifact before starting.
When useful, declare `execution_scale` as `pilot`, `direct_full`, or `full_after_pilot` without changing resource classification, for example: `SCIENCEFLOW_RESOURCE_VALUE_HINT='{{"execution_scale":"pilot","route_id":"stable_route_name"}}' python train.py`. Keep the same `route_id` when promoting a pilot to full scale.
During EDA, inventory all available input modalities/files. For each high-signal modality, either use it in a baseline branch or explicitly record why it is deferred.
Use a compact single-file solution for simple or short-run tasks. Consider reusable `train.py` / `predict.py` / `util.py` entrypoints only when training is expensive, validation/prediction should be repeated, or shared feature code prevents drift.
Keep training, feature transforms, prediction, and submission writing aligned through shared functions or shared modules so train/test feature logic does not drift.
If the current direction is exhausted or you would stop, reply with concise plain text and no tool call.
A valid full run should create root-level `submission.csv` and print `Final Validation Score: <float>` so the controller can record progress.
The reported validation score must be computed on validation rows that were not used for training, tuning, or ensemble-weight optimization; if you retrain on train+validation for final submission, print the held-out score from before that retrain. The reported held-out/CV score must come from the same validation split or folds used for final model selection; do not use an older or more optimistic split as a comparable final metric.
For small labeled datasets or tiny validation sets, treat one split as high-variance route evidence; use repeated/group CV, fold stability, or conservative OOF checks before trusting small holdout gains or repeatedly tuning weights, thresholds, or calibration on the same split.
Do not use host absolute paths; use workspace-relative scratch paths like `tmp/...`.
Use allocated compute deliberately. When ResourceContext or local probes show available GPU/CPU resources, treat sustained idle assigned compute as an execution issue to address with device placement, batching, parallelism, caching, data loading, model capacity, search breadth, ensembling, or inference throughput. If a route should remain resource-light, record the reason briefly.
{seed_block}{worker_identity_block}{skill_hint_block}Wall-clock budget: about {budget} seconds.{parallel_worker_block}{resource_context_block}{initial_workspace_state_block}
{runtime_contract_block}
Task:
{task}
