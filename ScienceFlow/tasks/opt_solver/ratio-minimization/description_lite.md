# Ratio minimization opt_solver task

This is a small mathematical optimization task adapted from public OpenEvolve / AlphaEvolve-style examples. It is not an MLEBench or Kaggle task.

## Problem

Place exactly 16 points in two-dimensional Euclidean space. The evaluator computes all pairwise distances.

A valid candidate must satisfy:

- the point array has shape `(16, 2)`;
- every coordinate is finite;
- the maximum pairwise distance is positive.

## Objective

The original minimization target is:

```text
max_pairwise_distance / min_pairwise_distance
```

ScienceFlow uses the equivalent maximization metric:

```text
inv_ratio_squared = (min_pairwise_distance / max_pairwise_distance)^2
```

Higher is better.

## Artifact

Write `artifacts/best_solution.json`:

```json
{"points": [[1.0, 0.0], [0.0, 1.0]]}
```

The list must contain exactly 16 two-dimensional points.

Use the configured task-package evaluator as the only authoritative score. Do not create `train.py` or `submission.csv`.
