# Circle packing opt_solver task

This is a small mathematical optimization task adapted from public OpenEvolve / AlphaEvolve-style examples. It is not an MLEBench or Kaggle task.

## Problem

Place exactly 26 circles inside the unit square `[0, 1] x [0, 1]`. Each circle has center `(x, y)` and radius `r`.

A valid candidate must satisfy:

- every coordinate and radius is finite;
- every radius is non-negative;
- every circle is fully inside the unit square;
- no two circles overlap, up to the evaluator tolerance in `dataset/problem.json`.

## Objective

Maximize the sum of all radii. The metric is `radii_sum`, higher is better.

## Artifact

Write `artifacts/best_solution.json`:

```json
{"circles": [[0.1, 0.1, 0.05], [0.2, 0.2, 0.04]]}
```

The list must contain exactly 26 `[x, y, radius]` triples.

Use the configured `artifact_command` evaluator as the only authoritative score. Do not create `train.py` or `submission.csv`.
