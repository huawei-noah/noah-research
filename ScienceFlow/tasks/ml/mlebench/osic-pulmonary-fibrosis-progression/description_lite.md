# Osic Pulmonary Fibrosis Progression — Lite Task Description

## Task description
Kaggle competition `osic-pulmonary-fibrosis-progression`.

## Task objective
- **Input:** `train.csv` has patient clinical records and full FVC history; `test.csv` has baseline clinical measurements. CT image folders may also be available.
- **Output:** Predict future FVC and uncertainty for each required `Patient_Week` row.

## Target metric (evaluation)
Modified Laplace Log Likelihood, higher is better. `Confidence` is clipped at a minimum of 70 by the scorer.

## Required validation protocol
- Use patient-level grouped validation. For each validation patient, expose only one baseline FVC measurement plus baseline clinical/CT features and keep the remaining FVC measurements as targets.
- Use only features available for a test patient at baseline; do not derive validation features such as slope, intercept, residual scale, or fitted trajectory from that patient's target history.
- Fit preprocessing, models, blend weights, and `Confidence` calibration on training patients only. Use inner patient-grouped splits for tuning, and judge routes by stable outer-fold performance rather than repeatedly optimized pooled OOF scores.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `Patient_Week,FVC,Confidence`; one row per required patient-week from `sample_submission.csv`.
- **Important:** `FVC` and `Confidence` must be numeric. Use the exact `Patient_Week` values from `sample_submission.csv`.
```
Patient_Week,FVC,Confidence
ID00126637202218610655908_-3,2000,100
ID00126637202218610655908_-2,2000,100
```

## Dataset and construction
- The authoritative runtime data directory is `dataset_split_v2`.
- `train.csv` contains 1,238 rows for 140 patients, with complete longitudinal histories. `validation.csv` contains 18 baseline-only rows for 18 patients, and `test.csv` contains 18 rows for 18 patients.
- The seed-42 split is grouped by patient: the train, validation, and test patient sets are mutually disjoint.
- `validation_sample_submission.csv` contains weeks -3 through 102 for every validation patient, for 1,908 rows total.
- CT data are in `train/ct`, `validation/ct`, and `test/ct`. Use each view only for Patient IDs present in its corresponding CSV.
- The 18-patient validation set is small; use inner patient-grouped folds to check stability.
