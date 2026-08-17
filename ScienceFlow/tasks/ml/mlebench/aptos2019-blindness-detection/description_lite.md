# Aptos2019 Blindness Detection — Lite Task Description

## Task description
Imagine being able to detect blindness before it happened.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Imagine being able to detect blindness before it happened.

## Target metric (evaluation)
Submissions are scored based on the quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** `id_code,diagnosis`; one row per test key, matching `sample_submission.csv`.
```
id_code,diagnosis
b460ca9fa26f,0
```

## Dataset and construction
- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
