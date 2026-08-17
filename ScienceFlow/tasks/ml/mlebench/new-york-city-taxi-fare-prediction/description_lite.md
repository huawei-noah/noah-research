# New York City Taxi Fare Prediction — Lite Task Description

## Task description
In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, this …

## Task objective
- **Input:** In the MLE-Bench prepared workspace, train from `labels.csv` and predict rows in `test.csv`; each row is identified by `key`.
- **Output:** For each `key` in the test set, you must predict a value for the `fare_amount` variable.

## Target metric (evaluation)
The evaluation metric for this competition is the root mean-squared error or RMSE. RMSE measures the difference between the predictions of a model, and the corresponding ground truth. A large RMSE is equivalent to a large average error, so smaller values of RMSE are better. One …

## RMSE tail guidance
RMSE squares large errors, so a small number of badly predicted tail trips can dominate the score. Do not optimize only average validation RMSE on ordinary short trips. Track validation slices for high-fare rides, airport or long-distance rides, very short or near-zero-distance rides, invalid or suspicious coordinates, year/time fare-regime changes, and surcharge/toll-like cases. A strong solution should combine a robust common-trip model with explicit tail handling or calibrated fallback rules.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Kernels-only**; see kernel rules.
- **Schema:** key,fare_amount; one row per test key (see `sample_submission.csv`).
```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
```

## Dataset and construction
- **labels.csv** - MLE-Bench prepared training data with input features and target `fare_amount` values. This is the actual train file in the public workspace; the original Kaggle text may refer to `train.csv`.
- **test.csv** - Input features for the held-out test rows. Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the … *(see full `description.md`.)*
- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in … *(see full `description.md`.)*

## Practical route
- Start with a leakage-safe holdout RMSE using cleaned coordinates, positive fares, valid passenger counts, and distance/time features.
- Strong baseline: LightGBM or XGBoost on haversine/Manhattan distance, coordinate deltas, bearing, pickup time parts, and coarse location/time bins.
- Useful upgrades: distances to stable landmarks or hubs, train-only grid/time aggregate features, and simple ensembles of diverse GBDT runs.
- Prioritize tail performance checks before accepting a model: compare RMSE on common trips versus high-fare, airport/long-distance, near-zero-distance, invalid-coordinate, and late-year/time slices.
- Treat exact/near-neighbor or target aggregate features as high-risk unless computed only from public training labels and validated out-of-fold.
