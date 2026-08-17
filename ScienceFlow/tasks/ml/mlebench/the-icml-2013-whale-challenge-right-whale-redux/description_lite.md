# The Icml 2013 Whale Challenge Right Whale Redux — Lite Task Description

## Task description
*(right whale illustration courtesy of Pieter Folkens, ©2011)* This competition complements the previously held Marinexplore Whale Detection Challenge, in which Cornell University provided data from a ship monitoring application termed "Auto Buoy", or AB Monitoring System. In the Marinexplore challenge we received …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** *(right whale illustration courtesy of Pieter Folkens, ©2011)*

## Target metric (evaluation)
Submissions are judged on area under the ROC curve. In Matlab (using the stats toolbox): ``` [~, ~, ~, auc ] = perfcurve(true_labels, predictions, 1); ``` In R (using the verification package): ``` auc = roc.area(true_labels, predictions) ``` In python (using the metrics module …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** clip,probability; one row per test key (see `sample_submission.csv`).
```
clip,probability
20090404_000000_012s0ms_Test0.aif,0
20090404_000000_042s6ms_Test1.aif,0
```

## Dataset and construction
- **train2.zip** - all of the training data as .aif clips. If the file ends in "_1.aif" it was labeled a whale call, if it ends in "_0.aif", it was labeled noise.
- **test2.zip** -  all of the testing data as .aif clips. You should predict a probability for each of these files.
