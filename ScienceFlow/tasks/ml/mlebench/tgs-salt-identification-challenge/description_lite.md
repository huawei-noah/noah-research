# Tgs Salt Identification Challenge — Lite Task Description

## Task description
Several areas of Earth with large accumulations of oil and gas *also* have huge deposits of salt below the surface. But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging still requires expert human interpretation of salt bodies. This leads to very subjective …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Predict targets for `tgs-salt-identification-challenge` test set per official rules.

## Target metric (evaluation)
This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as: *(formula in full description)* The metric sweeps over a range of IoU …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,rle_mask; one row per test key (see `sample_submission.csv`).
```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
```

## Dataset and construction
- ### Background Seismic data is collected using reflection seismology, or seismic reflection.
- The method requires a controlled seismic source of energy, such as compressed air or a seismic vibrator, and sensors record the reflection from rock interfaces within the subsurface.
- The recorded data is then processed to create a 3D view of earth's interior.
- Reflection seismology is similar to X-ray, sonar and echolocation.
