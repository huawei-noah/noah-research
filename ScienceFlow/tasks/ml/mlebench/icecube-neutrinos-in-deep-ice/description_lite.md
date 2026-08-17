# Icecube Neutrinos In Deep Ice — Lite Task Description

## Task description
Kaggle competition `icecube-neutrinos-in-deep-ice`.

## Task objective
- **Input:** Released `train` / `test` inputs (paths, IDs, features) as in prepared `public/`; labels only for train.
- **Output:** Predict targets for `icecube-neutrinos-in-deep-ice` test set per official rules.

## Target metric (evaluation)
Submissions are evaluated using the mean angular error between the predicted and true event origins. See this notebook for a python copy of the metric. Submission File --------------- For each `event_id` in the test set, you must predict the `azimuth` and `zenith`. The file …

## Brief background
One of the most abundant particles in the universe is the neutrino. While similar to an electron, the nearly massless and electrically neutral neutrinos have fundamental properties …

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** `event_id,azimuth,zenith`; one row per test key, matching `sample_submission.csv`.
```
event_id,azimuth,zenith
45566128,1,1
```

## Dataset and construction
- `batch_id` (`int`): the ID of the batch the event was placed into.
- `event_id` (`int`): the event ID.
- `[first/last]_pulse_index` (`int`): index of the first/last row in the features dataframe belonging to this event.
- `[azimuth/zenith]` (`float32`): the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. The target columns. Not provided for the test set. The direction … *(see full `description.md`.)*
