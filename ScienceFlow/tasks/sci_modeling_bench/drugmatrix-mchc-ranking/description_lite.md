# DrugMatrix MCHC Condition Ranking

Select and order 16 disclosed five-day treatment conditions expected to cause
the largest absolute control-relative change in mean corpuscular hemoglobin
concentration (MCHC, g/dL). The trusted score is
`abs(log(treatment_mean / matched_control_mean))`; a high score means a large
deviation, not a biologically desirable direction.

Read `dataset/dataset_manifest.json` for field semantics and units.
`dataset/views/observations.parquet` contains 9,442 individual-animal treatment
and matched-control observations for all six endpoints.
`dataset/views/candidates.parquet` contains 390 unlabeled treatment conditions
with molecule, dose, duration, vehicle, sex, route, and study context.
The candidate treatment-animal rows are withheld from the observations;
matched controls and other dose/time treatment rows remain visible.

Write at least 16 unique candidate IDs to `artifacts/submission.json`:

```json
{"candidates": [{"condition_id": "condition-id-from-candidate-view"}]}
```

Only the first 16 are scored; exactly 16 is recommended. The primary metric is
`global_ndcg`. Feedback includes `best_k_mean_regret`,
`normalized_enrichment`, and remaining queries, but no candidate endpoint
values. Four distinct lists may be evaluated.
