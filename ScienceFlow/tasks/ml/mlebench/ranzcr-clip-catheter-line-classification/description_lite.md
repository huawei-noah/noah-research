# Ranzcr Clip Catheter Line Classification — Lite Task Description

## Task description
The gold standard for the confirmation of line and tube positions are chest radiographs. However, a physician or radiologist must manually check these chest x-rays to verify that the lines and tubes are in the optimal position. Not only does this leave room for human error, but delays are also common as radiologists …

## Task objective
- **Input:** Each test row / notebook identified by `ID` (and any features in released `test` / `public` data).
- **Output:** For each ID in the test set, you must predict a probability for all target variables.

## Target metric (evaluation)
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target. To calculate the final score, AUC is calculated for each of the 11 labels, then averaged. The score is then the average of the individual AUCs of each predicted …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Required columns:** `StudyInstanceUID` plus all 11 target class columns required by the grader. Some prepared samples may omit trailing class columns, so use the grader class list below as the authority.
- **Target columns:** `ETT - Abnormal`, `ETT - Borderline`, `ETT - Normal`, `NGT - Abnormal`, `NGT - Borderline`, `NGT - Incompletely Imaged`, `NGT - Normal`, `CVC - Abnormal`, `CVC - Borderline`, `CVC - Normal`, `Swan Ganz Catheter Present`.
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset and construction
- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images
