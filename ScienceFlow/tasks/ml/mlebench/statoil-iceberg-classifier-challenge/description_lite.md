# Statoil Iceberg Classifier Challenge — Lite Task Description

## Task description
Statoil, an international energy company operating worldwide, has worked closely with companies like C-CORE. C-CORE have been using satellite data for over 30 years and have built a computer vision based surveillance system. To keep operations safe and efficient, Statoil is interested in getting a fresh new perspective …

## Task objective
- **Input:** Each test row identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1).

## Target metric (evaluation)
Submissions are evaluated on the log loss between the predicted values and the ground truth.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `id,is_iceberg`; one row per test key from `sample_submission.csv`.
- **Important:** `is_iceberg` must be a probability in `[0,1]`; clip or calibrate values before writing the final submission.
```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
```

## Dataset and construction
- **id** - the id of the image
- **band_1, band_2** - the flattened image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have … *(see full `description.md`.)*
- **inc_angle** - the incidence angle at which the image was taken. This field can be missing and marked as "na" in both train and test rows; handle it explicitly, and avoid treating missingness as a leakage shortcut.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.
