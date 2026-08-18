# Superconductor Measured-Pool Ranking

## Objective

Select and prioritize 32 measured material compositions expected to have high
critical temperature. This is ranking over the disclosed finite candidate
pool; do not generate compositions outside it.

## Agent-visible input

Read `dataset/dataset_manifest.json`, including the ordered `element_symbols`
context required to interpret composition vectors.
`dataset/views/observations.parquet` contains 16,795 labeled source
measurements from lower-temperature composition groups.
`dataset/views/candidates.parquet` contains 2,985 unlabeled normalized
composition vectors and representative formulas. The default setting does not
provide the published descriptor vector; construct chemical features from the
composition and element order.

## Submission

Write at least 32 distinct candidate compositions to
`artifacts/submission.json`:

```json
{"candidates": [{"composition": [0.0, 0.25, 0.0, 0.75]}]}
```

The short vector only illustrates the JSON field shape. A submitted vector
must have the full manifest-declared element length, contain
finite nonnegative fractions summing to one, and exactly match a candidate
pool composition. Only the first 32 candidates are scored; exactly 32 is
recommended.

## Evaluation

The primary metric is `global_ndcg`. Feedback also reports
`best_k_mean_regret`, `normalized_enrichment`, and remaining queries. It does
not reveal candidate temperatures. Ten distinct candidate lists may be
evaluated.
