# DrugMatrix Chloride Condition Ranking

Select and order 16 disclosed five-day treatment conditions expected to cause
the largest absolute control-relative change in chloride (mEq/L). The score is
`abs(log(treatment_mean / matched_control_mean))`; it measures magnitude in
either direction.

Inspect `dataset/dataset_manifest.json`, then use the 9,442 individual-animal
rows in `dataset/views/observations.parquet` and the 390 unlabeled conditions
in `dataset/views/candidates.parquet`. Candidate identity includes the frozen
treatment condition, not merely molecular structure.
The candidate treatment-animal rows are withheld from the observations;
matched controls and other dose/time treatment rows remain visible.

Write at least 16 unique IDs to `artifacts/submission.json`:

```json
{"candidates": [{"condition_id": "condition-id-from-candidate-view"}]}
```

Only the first 16 are scored. The primary metric is `global_ndcg`; feedback
also reports `best_k_mean_regret`, `normalized_enrichment`, and remaining
queries. Candidate-level values are hidden. Four distinct lists may be
evaluated.
