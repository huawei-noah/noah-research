# Smartphone Decimeter 2022 — Lite Task Description

## Task description
The goal of this competition is to compute smartphones location down to the decimeter or even centimeter resolution which could enable services that require lane-level accuracy such as HOV lane ETA estimation. You'll develop a model based on raw location measurements from Android smartphones collected in opensky and …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** For each `tripId` and `UnixTimeMillis` in the sample submission, predict `LatitudeDegrees` and `LongitudeDegrees`.

## Target metric (evaluation)
Submissions are scored on the mean of the 50th and 95th percentile distance errors. Lower is better. For every trip/device timestamp, the horizontal distance in meters is computed between predicted latitude/longitude and ground truth latitude/longitude. These distance errors form …

## Brief background
Have you ever missed the lane change before a highway exit? Do you want to know the estimated time of arrival (ETA) of a carpool lane rather than other lanes? These and other …

## Submission
- **File:** `submission.csv`.
- **Schema:** exactly match `sample_submission.csv`; one row per `(tripId, UnixTimeMillis)` test key.
```
tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees
2020-05-15-US-MTV-1_Pixel4,1273608785432,37.904611315634504,-86.48107806249548
2020-05-15-US-MTV-1_Pixel4,1273608786432,37.904611315634504,-86.48107806249548
```

## Local split notes
- The dataset split uses trip/device IDs rather than materialized train and validation directories.
- `train_trips.json` and `validation_trips.json` define the local train/validation split. They are disjoint and together cover the public train trip/device folders.
- Each split entry is `<drive_id>-<phone_name>`; read raw training data from `train/<drive_id>/<phone_name>/` after separating the phone suffix. The `train` path points to the public train directory; do not infer validation from test data.
- `test/` and `sample_submission.csv` define the required submission rows. Keep row keys and schema aligned with `sample_submission.csv`.

## Dataset and construction
- `MessageType` - "Fix", the prefix of sentence.
- `Provider` - "GT", short for ground truth.
- `[Latitude/Longitude]Degrees` - The WGS84 latitude, longitude (in decimal degrees) estimated by the reference GNSS receiver (NovAtel SPAN). When extracting from the NMEA file, linear interpolation has been applied to … *(see full `description.md`.)*
- `AltitudeMeters` - The height above the WGS84 ellipsoid (in meters) estimated by the reference GNSS receiver.
