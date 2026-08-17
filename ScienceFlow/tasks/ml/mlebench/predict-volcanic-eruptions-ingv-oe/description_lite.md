# Predict Volcanic Eruptions Ingv Oe — Lite Task Description

## Task description
What if scientists could anticipate volcanic eruptions as they predict the weather? While determining rain or shine days in advance is more difficult, weather reports become more accurate on shorter time scales. A similar approach with volcanoes could make a big impact. Just one unforeseen eruption can result in tens …

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** What if scientists could anticipate volcanic eruptions as they predict the weather? While determining rain or shine days in advance is more difficult, weather reports become more accurate on shorter time scales. A …

## Target metric (evaluation)
Submissions are evaluated on the mean absolute error (MAE) between the predicted loss and the actual loss.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** segment_id,time_to_eruption; one row per test key (see `sample_submission.csv`).
```
segment_id,time_to_eruption
1,1
2,2
```

## Dataset and construction
- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.
