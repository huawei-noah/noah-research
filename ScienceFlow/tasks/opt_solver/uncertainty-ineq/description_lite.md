# An uncertainty inequality opt_solver task

This is a small mathematical optimization task adapted from public OpenEvolve / AlphaEvolve-style examples. It is not an MLEBench or Kaggle task.

## Problem

Choose coefficients for an even Hermite-polynomial construction used in a Fourier-analysis uncertainty inequality. The evaluator constructs a polynomial from the submitted coefficients, chooses the final coefficient so that `P(0)=0`, recomputes the largest positive root of `P(x) / x^2`, and derives a `c4_bound`.

A valid candidate must satisfy:

- `coeffs` is a non-empty finite numeric list;
- the coefficient count does not exceed `dataset/problem.json:max_coeff_count`;
- the constructed polynomial yields a positive finite `c4_bound`;
- optional submitted `c4_bound` and `r_max` values match the evaluator recomputation.

## Objective

Maximize:

```text
c4_score = benchmark_c4 / computed_c4_bound
```

Higher is better. The evaluator recomputes the value and does not trust self-reported scores.

## Artifact

Write `artifacts/best_solution.json`:

```json
{"coeffs": [1.0, -0.25, 0.03], "c4_bound": 0.32, "r_max": 1.42}
```

Only `coeffs` is required. `c4_bound` and `r_max` are optional consistency checks.

Use the configured `artifact_command` evaluator as the only authoritative score. Do not create `train.py` or `submission.csv`.
