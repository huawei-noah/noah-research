# 3D Object Detection For Autonomous Vehicles — Lite Task Description

## Task description
Self-driving technology presents a rare opportunity to improve the quality of life in many of our communities. Avoidable collisions, single-occupant commuters, and vehicle emissions are choking cities, while infrastructure strains under rapid urban growth. Autonomous vehicles are expected to redefine transportation and …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Predict targets for `3d-object-detection-for-autonomous-vehicles` test set per official rules.

## Target metric (evaluation)
This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a set of predicted 3D bounding volumes and ground truth bounding volumes is calculated as: The metric sweeps over a range of IoU thresholds, at each point …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** Id,PredictionString; one row per test key (see `sample_submission.csv`).
```
Id,PredictionString
db8b47bd4ebdf3b3fb21598bb41bd8853d12f8d2ef25ce76edd4af4d04e49341,
97ce3ab08ccbc0baae0267cbf8d4da947e1f11ae1dbcb80c3f4408784cd9170c,1.0 2742.15 673.16 -18.65 1.834 4.609 1.648 2.619 car
```

## Dataset and construction
- `center_x`, `center_y` and `center_z` are the world coordinates of the center of the 3D bounding volume.
- `width`, `length` and `height` are the dimensions of the volume.
- `yaw` is the angle of the volume around the `z` axis (where `y` is forward/back, `x` is left/right, and `z` is up/down - making 'yaw' the direction the front of the vehicle / bounding box is pointing at while on the … *(see full `description.md`.)*
- `class_name` is the type of object contained by the bounding volume.
