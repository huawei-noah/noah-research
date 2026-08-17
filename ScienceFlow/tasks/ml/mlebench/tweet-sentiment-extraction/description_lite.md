# Tweet Sentiment Extraction — Lite Task Description

## Task description
*"My ridiculous dog is amazing."* [sentiment: positive] With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in …

## Task objective
- **Input:** Each test row / notebook identified by `ID` (and any features in released `test` / `public` data).
- **Output:** For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question.

## Target metric (evaluation)
The metric in this competition is the word-level Jaccard score. A good description of Jaccard similarity for strings is here. A Python implementation based on the links above, and matched with the output of the C# implementation on the back end, is provided below. ```python def …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** textID,selected_text; one row per test key (see `sample_submission.csv`).
```
textID,selected_text
2,"very good"
5,"I don't care"
```

## Dataset and construction
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- `textID` - unique ID for each piece of text
