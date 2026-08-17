# TFBind10 Pho4 Raw-Replicate Black-Box Optimization

## Objective

Propose 128 distinct DNA 10-mers predicted to have high Pho4 binding affinity.
This is free-form black-box optimization over the exhaustively measured
`4^10 = 1,048,576` sequence domain; there is no explicit candidate table.

## Agent-visible input

Read `dataset/dataset_manifest.json` for the frozen Dataset identity and full
field semantics. `dataset/views/observations.parquet` contains 2,087,323 raw
lower-half replicate rows with sequence, replicate identity, bound/input
counts, fractions, and `observed_ddg`. Several rows can therefore describe the
same sequence. Build sequence-level features or uncertainty estimates without
assuming that one raw replicate is a denoised label.

Each ten-character `sequence` stores the five variable bases immediately
upstream of the fixed `CACGTG` E-box core followed by the five variable bases
immediately downstream, all in the displayed strand's 5'-to-3' orientation.
The fixed core is not stored in the field; the full assayed site has the form
`NNNNNCACGTGNNNNN`.

The posterior affinity used for evaluation and the upper half of the sequence
landscape are not disclosed. The input does not contain an affinity score or a
candidate view.

## Submission

Write exactly 128 distinct candidates to `artifacts/submission.json`:

```json
{"candidates": [{"sequence": "AACCGGTTAA"}, {"sequence": "TTTTTTTTTT"}]}
```

Each sequence must be uppercase, length 10, and use only `A`, `C`, `G`, `T`.
Order the batch from highest to lowest predicted affinity.

## Evaluation

The primary metric is `normalized_enrichment` over the complete measured
domain. The secondary best-`K` summaries use `K=16`. Feedback also reports
`best_k_mean_regret`, `global_ndcg`, and the remaining query budget.
Candidate-level outcomes are never returned. Ten
distinct candidate batches may be evaluated; identical candidate content is
cached independently of JSON formatting.
