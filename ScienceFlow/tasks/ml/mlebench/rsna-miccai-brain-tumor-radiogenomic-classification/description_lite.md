# Rsna Miccai Brain Tumor Radiogenomic Classification — Lite Task Description

## Task description
Currently, genetic analysis of cancer requires surgery to extract a tissue sample. Then it can take several weeks to determine the genetic characterization of the tumor. Depending upon the results and type of initial therapy chosen, a subsequent surgery may be necessary. If an accurate method to predict the genetics of …

## Task objective
- **Input:** Each test row / notebook identified by `BraTS21ID` (and any features in released `test` / `public` data).
- **Output:** For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`.

## Target metric (evaluation)
Submissions are evaluated on the area under the ROC curve between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** BraTS21ID,MGMT_value; one row per test key (see `sample_submission.csv`).
```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
```

## Dataset and construction
- Fluid Attenuated Inversion Recovery (FLAIR)
- T1-weighted pre-contrast (T1w)
- T1-weighted post-contrast (T1Gd)
- T2-weighted (T2)
