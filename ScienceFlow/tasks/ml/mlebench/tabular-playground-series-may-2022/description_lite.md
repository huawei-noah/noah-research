# Tabular Playground Series May 2022 — Lite Task Description

## Task description
The May edition of the 2022 Tabular Playground series binary classification problem that includes a number of different feature interactions. This competition is an opportunity to explore various methods for identifying and exploiting these feature interactions.

## Task objective
- **Input:** Each test row / notebook identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each `id` in the test set, you must predict a probability for the `target` variable.

## Target metric (evaluation)
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,target; one row per test key (see `sample_submission.csv`).
```
id,target
900000,0.65
900001,0.97
```

## Dataset and construction
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format
