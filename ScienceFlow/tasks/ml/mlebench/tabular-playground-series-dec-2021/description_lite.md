# Tabular Playground Series Dec 2021 — Lite Task Description

## Task description
Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly. In order to have a …

## Task objective
- **Input:** Each test row / notebook identified by `Id` (and any features in released `test` / `public` data).
- **Output:** For each `Id` in the test set, you must predict the `Cover_Type` class.

## Target metric (evaluation)
Submissions are evaluated on multi-class classification accuracy.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** Id,Cover_Type; one row per test key (see `sample_submission.csv`).
```
Id,Cover_Type
4000000,2
4000001,1
```

## Dataset and construction
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format
