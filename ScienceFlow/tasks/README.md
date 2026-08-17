# Task metadata

- **`ml/mlebench/`** — MLE-bench/Kaggle-style task packages and shared evaluator.
- **`ml/mlebench/competition_categories.json`** — Maps a Kaggle-style competition id (`exp_id`) to a preset task category (for example, `Image Classification`). The LNR task-package resolver uses this table during cold start; update it when adding ML competitions.
- **`opt_solver/`** — Optimization/artifact task descriptions, including Luna Tomato Logistics and OpenEvolve-style math tasks.
- **`alg_opt/`** — Algorithm-theory research task packages. These may contain multi-stage controller templates in addition to the usual `task.yaml` and `description_lite.md`.
- **Per-task folders** — Each task package contains `task.yaml` plus `description_lite.md`; task-local `evaluator.py` is allowed when the provider has no shared evaluator.

If a competition id is **not** in the JSON, the solver does not inject a category skill; the agent relies on the task description and dataset preview only.
