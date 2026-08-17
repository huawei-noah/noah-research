# AI4Code — Lite Task Description

## Task description
Restore the full top-to-bottom order of cells in a Python Jupyter notebook. In the provided JSON, **code cells stay in the correct relative order**; **markdown cells are shuffled** and appended **after** all code cells. You must infer where each markdown cell belongs among the code cells.

## Task objective
- **Input:** Notebook identifier `id` (matches `{id}.json` in `train/` or `test/`).
- **Output:** `cell_order` — a single string of **space-separated cell IDs** (hex strings) listing **every** cell in `{id}.json` **exactly once**, in the **true** interleaved execution/read order (as in the original notebook).

## Target metric (evaluation)
**Kendall tau** over all test notebooks: for notebook *i* (*n_i* cells), *S_i* = adjacent swaps to match ground truth; **K = 1 − 4 × Σ_i S_i / Σ_i n_i(n_i − 1)** (higher better; use official reference implementation).

## Brief background
Public Kaggle Python notebooks; supervised **cell-sequence** reconstruction from code–markdown only.

## Submission
- **File:** `submission.csv` (Kaggle **Notebook / Code competition** submission path).
- **Schema:** header **`id,cell_order`**; one row per test notebook; `cell_order` = space-delimited cell IDs (same token set as in `{id}.json`).
```
id,cell_order
0009d135ece78d,ddfd239c c6cd22db 1372ae9b ...
```

## Dataset and construction
- **`train/`:** ~140K `{id}.json` — each notebook’s cells (code ordered, markdown shuffled to the tail).
- **`train_orders.csv`:** columns `id`, `cell_order` (ground truth).
- **`train_ancestors.csv`:** `ancestor_id`, `parent_id`, `id` — fork graph for **train only**; usable for group-wise validation (test notebooks have **no** train ancestor).
- **`test/`:** ~20K notebooks, same JSON layout, **no** labels.
- **`sample_submission.csv`:** required row set / column template.
