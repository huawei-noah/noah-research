# Hms Harmful Brain Activity Classification — Lite Task Description

## Task description
From stethoscopes to tongue depressors, doctors rely on many tools to treat their patients. Physicians use electroencephalography with critically ill patients to detect seizures and other types of brain activity that can cause brain damage. You can learn about how doctors interpret these EEG signals in these videos:\ …

## Task objective
- **Input:** Each test row / notebook identified by `eeg_id` (and any features in released `test` / `public` data).
- **Output:** For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns.

## Target metric (evaluation)
Submissions are evaluated on the Kullback Liebler divergence between the predicted probability and the observed target.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** eeg_id, … (remaining columns per sample_submission.csv); one row per test key (see `sample_submission.csv`).
```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
```

## Dataset and construction
- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
