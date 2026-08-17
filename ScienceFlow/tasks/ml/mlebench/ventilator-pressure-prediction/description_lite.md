# Ventilator Pressure Prediction — Lite Task Description

## Task description
What do doctors do when a patient has trouble breathing? They use a ventilator to pump oxygen into a sedated patient's lungs via a tube in the windpipe. But mechanical ventilation is a clinician-intensive procedure, a limitation that was prominently on display during the early days of the COVID-19 pandemic. At the same …

## Task objective
- **Input:** Each test row / notebook identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each `id` in the test set, you must predict a value for the `pressure` variable.

## Target metric (evaluation)
The competition will be scored as the mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored. The score is given by: *(formula in full description)* where is the vector of predicted pressure …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,pressure; one row per test key (see `sample_submission.csv`).
```
id,pressure
1,20
2,23
```

## Dataset and construction
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- `id` - globally-unique time step identifier across an entire file
