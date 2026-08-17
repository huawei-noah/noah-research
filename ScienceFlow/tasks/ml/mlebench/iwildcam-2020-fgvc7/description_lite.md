# Iwildcam 2020 Fgvc7 — Lite Task Description

## Task description
In order to tackle this problem, we have prepared a challenge where the training data and test data are from different cameras spread across the globe. The set of species seen in each camera overlap, but are not identical. The challenge is to classify species in the test cameras correctly. To explore multimodal …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** In order to tackle this problem, we have prepared a challenge where the training data and test data are from different cameras spread across the globe. The set of species seen in each camera overlap, but are not …

## Target metric (evaluation)
### Evaluation Submissions will be evaluated based on their categorization accuracy ### Submission Format The submission format for the competition is a csv file with the following format: ``` Id,Predicted 58857ccf-23d2-11e8-a6a3-ec086b02610b,1 …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `Id,Category`; one row per test key, matching `sample_submission.csv`.
```
Id,Category
879d74d8-21bc-11ea-a13a-137349068a90,559
```

## Dataset and construction
- ### Data Overview The WCS training set contains 217,959 images from 441 locations, and the WCS test set contains 62,894 images from 111 locations.
- These 552 locations are spread across the globe.
- You may also choose to use supplemental training data from the iNaturalist 2017, iNaturalist 2018 and iNaturalist 2019 competition datasets.
- As a courtesy, we have curated all the images from these datasets containing classes that might be in the test set and mapped them into the iWildCam categories.
