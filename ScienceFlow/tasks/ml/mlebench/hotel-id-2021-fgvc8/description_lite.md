# Hotel Id 2021 Fgvc8 — Lite Task Description

## Task description
Victims of human trafficking are often photographed in hotel rooms as in the below examples. Identifying these hotels is vital to these trafficking investigations but poses particular challenges due to low quality of images and uncommon camera angles.

## Task objective
- **Input:** Each test row / notebook identified by `image` (and any features in released `test` / `public` data).
- **Output:** For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image.

## Target metric (evaluation)
Submissions are evaluated according to the Mean Average Precision @ 5 (MAP@5): *(formula in full description)* where is the number of images, is the precision at cutoff , is the number of predictions per image, and is an indicator function equaling 1 if the item at rank is a …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** image,hotel_id; one row per test key (see `sample_submission.csv`).
```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
```

## Dataset and construction
- `image` - The image ID.
- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per … *(see full `description.md`.)*
- `hotel_id` - The hotel ID. The target class.
- `timestamp` - When the image was taken. Provided for the training set only.
