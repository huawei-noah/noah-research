# DrugMatrix Phosphorus Condition Ranking

Select and order 16 disclosed five-day treatment conditions expected to cause
the largest absolute control-relative change in phosphorus (mg/dL). The score
is `abs(log(treatment_mean / matched_control_mean))`, so it rewards deviation
magnitude rather than increase, decrease, or clinical desirability.

Read `dataset/dataset_manifest.json` for the exact field meanings and units.
`dataset/views/observations.parquet` contains 9,442 individual-animal rows for
all endpoints. `dataset/views/candidates.parquet` contains 390 unlabeled
conditions with molecule and experimental context.
The candidate treatment-animal rows are withheld from the observations;
matched controls and other dose/time treatment rows remain visible.

Write at least 16 unique IDs to `artifacts/submission.json`:

```json
{"candidates": [{"condition_id": "condition-id-from-candidate-view"}]}
```

Only the first 16 are scored. `global_ndcg` is primary; feedback includes
`best_k_mean_regret`, `normalized_enrichment`, and remaining queries without
candidate truth. Four distinct lists may be evaluated.
