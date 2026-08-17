# Alaska2 Image Steganalysis — Lite Task Description

## Task description
That file you downloaded may contain hidden messages that aren't part of its regular contents. The same technology employed for digital watermarking is also misused by crime rings. Law enforcement must now use steganalysis to detect these messages as part of their investigations. Machine learning is an important tool …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data.

## Target metric (evaluation)
Submissions are evaluated by **weighted AUC** and **higher is better**. The metric emphasizes reliable detection at low false-alarm rates. It uses TPR bands `[0.0, 0.4, 1.0]` with heavier weight on the low-FPR / early-TPR region, so plain ROC AUC can be misleading. Select models by the official weighted AUC or a faithful local implementation, not by standard AUC alone.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** Id,Label; one row per test key (see `sample_submission.csv`).
```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
```

## Dataset and construction
- The prepared Deep v2 data contains 70,000 complete source-image groups. Each basename identifies one group with four aligned files: one image in `Cover/` and one corresponding image in each of `JMiPOD/`, `JUNIWARD/`, and `UERD/`.
- Manifest labels are binary: the `Cover/` row is label `0`, while the three steganography rows are label `1`. The four rows sharing a basename are paired variants of the same source image, not independent source images.
- Use the provided `train.jsonl` and `validation.jsonl` split. Do not create a row-wise random split; any alternative split or cross-validation must keep all rows sharing a basename in the same fold.
- The provided split has 252,000 training rows from 63,000 source groups and 28,000 validation rows from 7,000 disjoint source groups.
- `Test/` contains 5,000 images. `sample_submission.csv` defines the required test IDs and submission schema.
