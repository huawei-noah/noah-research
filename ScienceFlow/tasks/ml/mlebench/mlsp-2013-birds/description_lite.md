# Mlsp 2013 Birds — Lite Task Description

## Task description
It is important to gain a better understanding of bird behavior and population trends. Birds respond quickly to environmental change, and may also tell us about other organisms (e.g., insects they feed on), while being easier to detect. Traditional methods for collecting data about birds involve costly human effort. A …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** It is important to gain a better understanding of bird behavior and population trends. Birds respond quickly to environmental change, and may also tell us about other organisms (e.g., insects they feed on), while being …

## Target metric (evaluation)
Submissions are judged on area under the ROC curve. In Matlab (using the stats toolbox): ``` [~, ~, ~, auc ] = perfcurve(true_labels, predictions, 1); ``` In R (using the verification package): ``` auc = roc.area(true_labels, predictions) ``` In python (using the metrics module …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `Id,Probability`; one row per test key, matching `sample_submission.csv`.
```
Id,Probability
100,0
```

## Dataset and construction
- mlsp_contest_dataset.zip - Contains all necessary and supplemental files for the competition + additional documentation.
- mlsp13birdchallenge_documentation.pdf - Main dataset documentation. Has more info than what is on the site.
- ** Essential Files ***
