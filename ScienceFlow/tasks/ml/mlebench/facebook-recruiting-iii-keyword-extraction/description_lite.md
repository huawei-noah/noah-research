# Facebook Recruiting Iii Keyword Extraction — Lite Task Description

## Task description
Looking for a data science position at Facebook? After two successful prior Kaggle competitions, Facebook continues their mission to identify the best data scientists and software engineers that Kaggle has to offer. In this third installment, they seek candidates who have experience text mining large amounts of data.

## Task objective
- **Input:** `train.csv` has `Id,Title,Body,Tags`; `test.csv` has `Id,Title,Body`.
- **Output:** Predict Stack Exchange keyword tags for every test question.

## Target metric (evaluation)
The evaluation metric for this competition is Mean F1-Score. The F1 score, commonly used in information retrieval, measures accuracy using the statistics precision p and recall r. Precision is the ratio of true positives (tp) to all predicted positives (tp + fp). Recall is the …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Required columns:** `Id,Tags`; one row per test question. The prepared sample may include extra text columns, but the grader requires `Id` and `Tags`.
- **Important:** `Tags` must be a space-delimited string of predicted tags, not a generic `prediction` column.
- **Do not leave `Tags` blank:** every row must contain a non-empty string. Empty strings, missing values, `NaN`, or null values are invalid; use a valid fallback tag string if no model tag is confident.
- **Keep only submission columns:** final `submission.csv` should contain `Id,Tags` only, even though `sample_submission.csv` may include `Title` and `Body`.
```
Id,Tags
1,"c++ javascript"
2,"php python mysql"
```

## Dataset and construction
- Id - Unique identifier for each question
- Title - The question's title
- Body - The body of the question
- Tags - The tags associated with the question (all lowercase, should not contain tabs '\t' or ampersands '&')
