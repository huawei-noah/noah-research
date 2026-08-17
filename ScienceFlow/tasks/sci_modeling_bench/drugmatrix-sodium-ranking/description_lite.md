# DrugMatrix Sodium Condition Ranking

Select and order 16 disclosed five-day treatment conditions expected to cause
the largest absolute control-relative change in sodium (mEq/L). The trusted
score is `abs(log(treatment_mean / matched_control_mean))`; high score means a
large change in either direction, not desirable sodium concentration.

Read `dataset/dataset_manifest.json`. The 9,442-row observations view contains
individual-animal measurements for all six endpoints. The 390-row candidates
view contains unlabeled chemical conditions with dose, duration, vehicle, sex,
route, and study context. Source measurements retain real noise and unusual
extremes; do not silently assume that all outliers are data-entry errors.
The candidate treatment-animal rows are withheld from the observations;
matched controls and other dose/time treatment rows remain visible.

Write at least 16 unique `condition_id` mappings to
`artifacts/submission.json`:

```json
{"candidates": [{"condition_id": "condition-id-from-candidate-view"}]}
```

Only the first 16 are scored; exactly 16 is recommended. `global_ndcg` is
primary. Feedback adds
`best_k_mean_regret`, `normalized_enrichment`, and remaining queries without
revealing candidate truth. Four distinct lists may be evaluated.
