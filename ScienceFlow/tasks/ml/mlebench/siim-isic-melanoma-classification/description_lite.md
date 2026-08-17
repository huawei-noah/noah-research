# Siim Isic Melanoma Classification — Lite Task Description

## Task description
Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma. Existing AI approaches have not adequately considered this clinical frame of reference. Dermatologists could enhance their diagnostic accuracy if detection algorithms …

## Task objective
- **Input:** Each test row / notebook identified by `image_name` (and any features in released `test` / `public` data).
- **Output:** For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**.

## Target metric (evaluation)
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** image_name,target; one row per test key (see `sample_submission.csv`).
```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
```

## Dataset and construction
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- `image_name` - unique identifier, points to filename of related DICOM image
