# UTR MRL Compositional Candidate Ranking

## Objective

Select and order 128 measured 50-nucleotide 5' UTR candidates predicted to
have high mean ribosome load (MRL) in the held-out biological combination.
This is ranking over an explicit finite candidate pool, not generation over
the open `4^50` sequence space.

## Agent-visible input

`dataset/dataset_manifest.json` defines the two disclosed views and every
field. `dataset/views/observations.parquet` contains 76,877 labeled sequences
from three visible uAUG/Kozak combinations. `dataset/views/candidates.parquet`
contains 5,043 unlabeled sequences from the held-out combination. The per-row
partition annotations are hidden; derive motif, frame, composition, or
structure features from sequence when useful.

Each `sequence` is the DNA-alphabet representation of the 50-nucleotide
variable segment at the 3' end of the reporter 5' leader. A fixed
25-nucleotide leader segment precedes it and the fixed eGFP main start codon
`ATG` follows it immediately; `T` represents `U` in the transcribed RNA.

## Submission

Write an ordered candidate list to `artifacts/submission.json`:

```json
{"candidates": [{"sequence": "ACGT...50 bases..."}]}
```

The placeholder is schematic and is not a valid sequence. Submit at least 128
unique complete sequences from the disclosed candidate view. Only
the first 128 are scored, so exactly 128 is recommended. Invalid, repeated, or
out-of-pool sequences make the scored prefix ineligible.

## Evaluation

The primary metric is `normalized_enrichment`, which measures the quality of
the complete 128-sequence batch relative to random pool selection and the
measured top-128 batch. The secondary best-`K` summaries use `K=16`.
Feedback also includes `best_k_mean_regret`, `global_ndcg`, and the remaining
budget. Candidate-level measurements remain hidden. Five distinct batches may
be evaluated.
