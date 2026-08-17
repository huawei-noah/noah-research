# Vinbigdata Chest Xray Abnormalities Detection — Lite Task Description

## Task description
Existing methods of interpreting chest X-ray images classify them into a list of findings. There is currently no specification of their locations on the image which sometimes leads to inexplicable results. A solution for localizing findings on chest X-ray images is needed for providing doctors with more meaningful …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Existing methods of interpreting chest X-ray images classify them into a list of findings. There is currently no specification of their locations on the image which sometimes leads to inexplicable results. A solution for …

## Target metric (evaluation)
The challenge uses the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.4.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** image_id,PredictionString; one row per test key (see `sample_submission.csv`).
```
image_id,PredictionString
004f33259ee4aef671c2b95d54e4be68,14 1 0 0 1 1
004f33259ee4aef671c2b95d54e4be69,11 0.5 100 100 200 200 13 0.7 10 10 20 20
```

## Dataset and construction
- **train.csv** - the train set metadata, with one row for each object, including a class and a bounding box. Some images in both test and train have multiple objects.
- **sample_submission.csv** - a sample submission file in the correct format
- `image_id` - unique image identifier
- `class_name` - the name of the class of detected object (or "No finding")
