# H And M Personalized Fashion Recommendations — Lite Task Description

## Task description
H&M Group is a family of brands and businesses with 53 online markets and approximately 4,850 stores. Our online store offers shoppers an extensive selection of products to browse through. But with too many choices, customers might not quickly find what interests them or what they are looking for, and ultimately, they …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** For each `customer_id` observed in the training data, predict up to 12 `article_id` values for the next 7-day purchase period. Keep `article_id` values as 10-character strings with leading zeroes.

## Target metric (evaluation)
Submissions are evaluated according to the Mean Average Precision @ 12 (MAP@12): *(formula in full description)* where is the number of customers, is the precision at cutoff is the number predictions per customer, is the number of ground truth values per customer, and is an …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** customer_id,prediction; one row per test key (see `sample_submission.csv`).
```
customer_id,prediction
00000dba,0706016001 0706016002 0372860001 ...
0000423b,0706016001 0706016002 0372860001 ...
```

## Dataset and construction
- **images/** - a folder of images corresponding to each `article_id`; images are placed in subfolders starting with the first three digits of the `article_id`; note, not all `article_id` values have a corresponding image.
- **articles.csv** - detailed metadata for each `article_id` available for purchase
- **customers.csv** - metadata for each `customer_id` in dataset
- **sample_submission.csv** - a sample submission file in the correct format
- `dataset_split_v2` is a time-based split: train uses earlier transactions, `validation.csv` uses the later validation window, and `validation_answers.csv` gives grouped MAP@12 targets.
- This is a time-ordered recommender task where generalization across future periods matters, so avoid optimizing only for the single provided validation window.
- When feasible, build earlier rolling time validation windows from `transactions_train.csv` and prefer approaches that remain stable across multiple windows.
