# Chaii Hindi And Tamil Question Answering — Lite Task Description

## Task description
In this competition, your goal is to predict answers to real questions about Wikipedia articles. You will use chaii-1, a new question answering dataset with question-answer pairs. The dataset covers Hindi and Tamil, collected without the use of translation. It provides a realistic information-seeking task with …

## Task objective
- **Input:** Each test row / notebook identified by `ID` (and any features in released `test` / `public` data).
- **Output:** For each ID in the test set, you must predict the string that best answers the provided question based on the context.

## Target metric (evaluation)
The metric in this competition is the word-level Jaccard score. A good description of Jaccard similarity for strings is here. A Python implementation based on the links above, and matched with the output of the C# implementation on the back end, is provided below. ```python def …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** id,PredictionString; one row per test key (see `sample_submission.csv`).
```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
```

## Dataset and construction
- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format
- `id` - a unique identifier
