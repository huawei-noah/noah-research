# Cassava Leaf Disease Classification — Lite Task Description

## Task description
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa …

## Target metric (evaluation)
Categorization accuracy; higher is better. Report `Final Validation Score` as raw validation accuracy and set `lower_is_better=false`.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `image_id,label`; one row per test key, matching `sample_submission.csv`.
```
image_id,label
1234294272.jpg,4
```

## Dataset and construction
- `image_id` the image file name.
- `label` the ID code for the disease.
- `image_id` the image file name.
- `label` the predicted ID code for the disease.
