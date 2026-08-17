# DrugMatrix Creatinine Condition Ranking

Select and order 16 disclosed five-day treatment conditions expected to cause
the largest absolute control-relative change in creatinine (mg/dL). The score
is `abs(log(treatment_mean / matched_control_mean))`; it measures deviation
magnitude, not whether creatinine increases or decreases.

Read `dataset/dataset_manifest.json`. The 9,442-row
`dataset/views/observations.parquet` table contains individual animals and all
six endpoints. `dataset/views/candidates.parquet` contains 390 unlabeled
conditions with molecule, dose, duration, vehicle, sex, route, and study.
The candidate treatment-animal rows are withheld from the observations;
matched controls and other dose/time treatment rows remain visible.

Write at least 16 unique IDs to `artifacts/submission.json`:

```json
{"candidates": [{"condition_id": "condition-id-from-candidate-view"}]}
```

Only the first 16 are scored. The primary metric is `global_ndcg`; feedback
also reports `best_k_mean_regret`, `normalized_enrichment`, and remaining
queries. Candidate endpoint values remain hidden. Four distinct lists may be
evaluated.
