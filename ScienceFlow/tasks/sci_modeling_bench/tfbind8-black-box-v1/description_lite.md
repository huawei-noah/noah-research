# TFBind8 Offline Black-Box Design

Design 128 DNA sequences that maximize the TFBind8 normalized E-score. The available
observations are in `dataset/offline_data.csv`; they contain only the bottom-half
offline view defined by the benchmark protocol.

Write the final candidate batch to `artifacts/submission.json`:

```json
{
  "candidates": [
    {"sequence": "AACCGGTT"},
    {"sequence": "TTTTTTTT"}
  ]
}
```

Submission contract:

- `candidates` must contain exactly 128 entries.
- Each sequence must be an uppercase DNA string of length 8.
- The only allowed symbols are `A`, `C`, `G`, and `T`.
- Duplicate candidates are allowed but do not improve the top-1 score.
- The primary metric is the maximum `normalized_e_score` among valid candidates.
- Higher is better.

The complete TFBind8 table and exact objective are evaluator-only. Do not treat local
surrogate or cross-validation scores as the authoritative result.
