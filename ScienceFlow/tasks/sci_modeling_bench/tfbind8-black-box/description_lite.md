# TFBind8 Offline Black-Box Optimization

## Objective

Propose 32 distinct uppercase DNA 8-mers with high normalized TFBind8
E-score. Candidates may be any legal sequence in the complete `4^8 = 65,536`
domain; they are not restricted to the observed rows.

## Agent-visible input

Read `dataset/dataset_manifest.json` first. It records the pinned Dataset and
Protocol and gives the scientific meaning, type, unit, and constraints of every
visible field. `dataset/views/observations.parquet` contains 32,768 unique
lower-half observations with `sequence` and `normalized_e_score` columns.
`dataset/dataset_files.json` maps the manifest view to its Parquet file, and
`dataset/task_contract.json` records the submission and feedback contract.

The complete landscape and exact candidate outcomes are evaluator-only.

## Submission

Write exactly 32 ordered candidates to `artifacts/submission.json`:

```json
{"candidates": [{"sequence": "AACCGGTT"}, {"sequence": "TTTTTTTT"}]}
```

Every sequence must have length 8, use only `A`, `C`, `G`, and `T`, and be
distinct. Order candidates from highest to lowest predicted quality.

## Evaluation

The primary metric is `best_k_mean`, the mean trusted score of the five best
candidates anywhere in the batch. Higher is better. Feedback also reports
`best_k_mean_regret`, `global_ndcg`, and the remaining query budget. It never
reveals per-candidate objective values.

At most 10 distinct candidate batches receive official evaluation. A
semantically identical candidate list reuses its cached result even if JSON
whitespace or key order changes.
