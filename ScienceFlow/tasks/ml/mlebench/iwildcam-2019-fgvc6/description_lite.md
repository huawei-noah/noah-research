# Iwildcam 2019 Fgvc6 — Lite Task Description

## Task description
Camera Traps (or Wild Cams) enable the automatic collection of large quantities of image data. Biologists all over the world use camera traps to monitor biodiversity and population density of animal species. We have recently been making strides towards automating the species classification challenge in camera traps …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Camera Traps (or Wild Cams) enable the automatic collection of large quantities of image data. Biologists all over the world use camera traps to monitor biodiversity and population density of animal species. We have …

## Target metric (evaluation)
### Evaluation Submissions will be evaluated based on their macro F1 score - i.e. F1 will be calculated for each class of animal (including "empty" if no animal is present), and the submission's final score will be the unweighted mean of all class F1 scores.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `,Id,Category`; one row per test key, matching `sample_submission.csv`.
```
,Id,Category
0,5998cfa4-23d2-11e8-a6a3-ec086b02610b,0
```

## Dataset and construction
- ### Data Overview The training set contains 196,157 images from 138 different locations in Southern California.
- You may also choose to use supplemental training data from iNaturalist 2017, iNaturalist 2018, iNaturalist 2019, and images simulated with Microsoft AirSim.
- As a courtesy, we have curated all the images from iNaturalist 2017/2018 containing classes that might be in the test set and mapped them into the iWildCam categories.
- This data (which we call "iNat Idaho") can be downloaded from our git page here.
