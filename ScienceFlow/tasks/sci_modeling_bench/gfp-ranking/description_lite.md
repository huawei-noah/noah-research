# GFP Measured-Protein Candidate Ranking

## Objective

Select and order 128 measured GFP protein variants predicted to have high
median log10 brightness. This is finite-pool ranking at protein identity, not
generation over arbitrary proteins.

## Agent-visible input

Read `dataset/dataset_manifest.json`; it describes four views and the
237-residue reference sequence context.

- `dataset/views/protein_observations.parquet`: 41,372 labeled protein rows;
- `dataset/views/nucleotide_observations.parquet`: 42,943 visible synonymous
  nucleotide aggregates;
- `dataset/views/barcode_observations.parquet`: 45,577 visible barcode
  measurements and coverage context;
- `dataset/views/candidates.parquet`: 10,343 unlabeled measured protein
  variants with sequence-derived fields.

Candidate brightness, barcode coverage, dispersion, and nucleotide-level
measurements are not exposed through a hidden join.

## Submission

Write at least 128 unique candidate sequences to `artifacts/submission.json`:

```json
{"candidates": [{"sequence": "MSKGEELF..."}]}
```

The ellipsis is schematic and is not a valid submission value. Every submitted
sequence must be a complete 237-residue sequence present in
the candidate view. Only the first 128 are scored; exactly 128 is recommended.

## Evaluation

The primary metric is `normalized_enrichment`, which measures complete-batch
brightness relative to random pool selection and the measured top-128 batch.
The secondary best-`K` summaries use `K=16`. Feedback additionally reports
`best_k_mean_regret`, `global_ndcg`, and the remaining query budget.
Per-protein brightness is never revealed. Ten distinct lists may be
evaluated.
