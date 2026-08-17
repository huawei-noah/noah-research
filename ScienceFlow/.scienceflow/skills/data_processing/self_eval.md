---
name: self_eval
category: data_processing
tags: [evaluation, workspace, leakage-prevention, flat-dataset]
task_types: ["*"]
phase: [data_preparation, evaluation]
priority: 90
executable: true
max_retries: 5
context_vars: [run_data_dir, task_desc, readonly_warning, selftest_error]
---

# `workspace_eval.py` Generation

Generate one local evaluation module at `dataset/workspace_eval.py`. Do **not** create or write Deep/Shallow split directories.

Use multiple ```python / ```bash blocks across turns if needed.

## Task

{task_desc}

{selftest_error}

## Framework note (read-only)

{readonly_warning}

## Dataset root (`run_data_dir`)

The framework passes **`{run_data_dir}`**: the absolute path to the flat dataset root, usually `<task_workspace>/workspace/dataset`.

Set `DATASET_ROOT = Path(r"{run_data_dir}")` at the top. Discover the actual files with `iterdir`, `rglob`, `os.walk`, or small `pd.read_csv(..., nrows=5)` probes before writing the evaluator.

## Expected Inputs

Prefer these files when present:

- `validation*` files under `DATASET_ROOT` for validation features/rows.
- `mask_validation_label.*` under `DATASET_ROOT` for hidden validation labels.
- Existing train metadata when a local holdout must be reconstructed.

## Required Output

Write `DATASET_ROOT / "workspace_eval.py"` with:

- `eval(preds)` as the primary function.
- Optional `evaluate(...)` wrapper if useful.
- A small `__main__` or self-test path that can run without a real submission file.
- A printed `METRIC_VALUE=<float>` line during self-test when possible.

## Hard Rules

1. Write only `workspace_eval.py` inside `DATASET_ROOT`.
2. Do not move, overwrite, or recreate validation files or hidden labels.
3. Do not read test labels or external prepared directories.
4. Use the exact row order expected by the validation files.
5. Keep the first evaluator simple and executable; refine the metric only after the file exists and self-test passes.
