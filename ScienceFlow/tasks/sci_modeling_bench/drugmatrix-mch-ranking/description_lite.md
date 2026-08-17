# DrugMatrix MCH Condition Ranking

Select and order 16 disclosed five-day treatment conditions expected to cause
the largest absolute control-relative change in mean corpuscular hemoglobin
(MCH, pg). The trusted score is
`abs(log(treatment_mean / matched_control_mean))`; it measures deviation
magnitude rather than a desirable clinical direction.

Read `dataset/dataset_manifest.json` for field semantics and units.
`dataset/views/observations.parquet` contains 9,442 individual-animal rows for
all six endpoints, and `dataset/views/candidates.parquet` contains 390
unlabeled conditions with chemical and experimental context.
The candidate treatment-animal rows are withheld from the observations;
matched controls and other dose/time treatment rows remain visible.

Write at least 16 unique IDs to `artifacts/submission.json`:

```json
{"candidates": [{"condition_id": "condition-id-from-candidate-view"}]}
```

Only the first 16 are scored; exactly 16 is recommended. `global_ndcg` is the
primary metric. Feedback includes `best_k_mean_regret`,
`normalized_enrichment`, and remaining queries without candidate-level truth.
Four distinct lists may be evaluated.
