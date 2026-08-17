# Text Normalization Challenge Russian Language — Lite Task Description

## Task description
As many of us can attest, learning another language is tough. Picking up on nuances like slang, dates and times, and local expressions, can often be a distinguishing factor between proficiency and fluency. This challenge is even more difficult for a machine. Many speech and language applications, including …

## Task objective
- **Input:** Each test row / notebook identified by `token (id)` (and any features in released `test` / `public` data).
- **Output:** For each token (id) in the test set, you must predict the normalized text.

## Target metric (evaluation)
Submissions are evaluated on prediction accuracy (the total percent of correct tokens). The predicted and actual string must match exactly in order to count as correct. In other words, we are measuring sequence accuracy, in that any error in the output for a given token in the …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,after; one row per test key (see `sample_submission.csv`).
```
id,after
0_0,"Производится"
0_1,"в"
```

## Dataset and construction
- ru_sample_submission.csv - a submission file showing the correct format
- ru_test.csv - the test set, does not contain the normalized text
- ru_train.csv - the training set, contains the normalized text
