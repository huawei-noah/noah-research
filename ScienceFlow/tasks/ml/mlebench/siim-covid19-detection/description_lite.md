# Siim Covid19 Detection — Lite Task Description

## Task description
Five times more deadly than the flu, COVID-19 causes significant morbidity and mortality. Like other pneumonias, pulmonary infection with COVID-19 results in inflammation and fluid in the lungs. COVID-19 looks very similar to other viral and bacterial pneumonias on chest radiographs, which makes it difficult to …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Five times more deadly than the flu, COVID-19 causes significant morbidity and mortality. Like other pneumonias, pulmonary infection with COVID-19 results in inflammation and fluid in the lungs. COVID-19 looks very …

## Target metric (evaluation)
The challenge uses the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > `0.5`. Note that the linked document describes VOC 2012, which differs in some minor ways (e.g. there is no concept of "difficult" classes in VOC 2010). The P/R curve and AP calculations remain …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** Id,PredictionString; one row per test key (see `sample_submission.csv`).
```
Id,PredictionString
2b95d54e4be65_study,negative 1 0 0 1 1
2b95d54e4be66_study,typical 1 0 0 1 1
```

## Dataset and construction
- **train_study_level.csv** - the train study-level metadata, with one row for each study, including correct labels.
- **train_image_level.csv** - the train image-level metadata, with one row for each image, including both correct labels and any bounding boxes in a dictionary format. Some images in both test and train have multiple … *(see full `description.md`.)*
- **sample_submission.csv** - a sample submission file containing all image- and study-level IDs.
- `id` - unique study identifier
