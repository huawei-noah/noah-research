# Spooky Author Identification — Lite Task Description

## Task description
In this year's Halloween playground competition, you're challenged to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft. We're encouraging you (with cash prizes!) to share your insights in the competition's discussion forum and code in Kernels. We've designated prizes …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** You must submit a csv file with the id, and a probability for each of the three classes.

## Target metric (evaluation)
Submissions are evaluated using multi-class logarithmic loss. Each id has one true class. For each id, you must submit a predicted probability for each author. The formula is then: *(formula in full description)* where N is the number of observations in the test set, M is the …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id, … (remaining columns per sample_submission.csv); one row per test key (see `sample_submission.csv`).
```
id,EAP,HPL,MWS
id07943,0.33,0.33,0.33
...
```

## Dataset and construction
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **id** - a unique identifier for each sentence
