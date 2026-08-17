# Multi Modal Gesture Recognition — Lite Task Description

## Task description
The Multi-modal gesture recognition challenge, focused on gesture recognition from 2D and 3D video data using Kinect, is organized by ChaLearn in conjunction with ICMI 2013. Kinect is revolutionizing the field of gesture recognition given the set of input data modalities it provides, including RGB image, depth image …

## Task objective
- **Input:** Multi-modal Kinect gesture data for train/test sessions; labels only for train.
- **Output:** Predict the ordered gesture sequence for each test session.

## Target metric (evaluation)
Submissions are evaluated on the Levenshtein edit distance between the predicted gesture label sequence and the ground-truth sequence, averaged over test sessions (lower is better). The challenge uses a "multiple instance, user independent" gesture vocabulary of 20 categories.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** `Id,Sequence`; one row per test session, matching `randomPredictions.csv` when `sample_submission.csv` is absent.
- **Important:** `Sequence` must be a space-delimited list of integer gesture labels.
```
Id,Sequence
0300,13 14 2 9 16 7 20 5
```

## Dataset and construction
- SessionID_audio: Audio file.
- SessionID_color: Video file with the RGB information.
- SessionID_depth: Video file with the Depth information
- SessionID_user: Video file with the user segmentation information
