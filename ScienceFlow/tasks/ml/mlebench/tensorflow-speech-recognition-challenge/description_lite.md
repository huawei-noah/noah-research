# Tensorflow Speech Recognition Challenge — Lite Task Description

## Task description
We might be on the verge of too many screens. It seems like everyday, new versions of common objects are “re-invented” with built-in wifi and bright touchscreens. A promising antidote to our screen addiction are voice interfaces.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** For each audio file name, predict one of the allowed speech command labels.

## Target metric (evaluation)
Submissions are evaluated on Multiclass Accuracy, which is simply the average number of observations with the correct label. **Note:** There are only 12 possible labels for the Test set: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`, `silence`, `unknown`. …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** fname,label; one row per test key (see `sample_submission.csv`).
```
fname,label
clip_000044442.wav,silence
clip_0000adecb.wav,left
```

## Dataset and construction
- **train/** - Audio files grouped by label, with split metadata also available in `train.csv`.
- **test/** - Test audio files in the format `clip_000044442.wav`; predict the correct label for each row in `sample_submission.csv`.
- **sample_submission.csv** - A sample submission file in the correct format.
