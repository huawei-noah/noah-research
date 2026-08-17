# Detecting Insults In Social Commentary — Lite Task Description

## Task description
The challenge is to detect when a comment from a conversation would be considered insulting to another participant in the conversation. Samples could be drawn from conversation streams like news commenting sites, magazine comments, message boards, blogs, text messages, etc.

## Task objective
- **Input:** Comment text and optional timestamp metadata; labels only for train.
- **Output:** Predict insult probability for each test comment.

## Target metric (evaluation)
This is a single-class classification problems. Your predictions should be a number in the range [0,1] where 0 indicates 0% probability of comment being an insult, and 1 represents 100% insult. All predictions should be in the first column of your submission file. Please see …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` (match the prepared `sample_submission_null.csv` layout when present).
- **Required columns:** `Comment,Insult`; the sample also includes `Date`. The grader matches rows by exact `Comment` text and scores the `Insult` probability.
- **Important:** `Insult` must be numeric in `[0,1]`; do not use a generic `prediction` column.
```
Insult,Date,Comment
0,,"example comment text"
```

## Dataset and construction
- We are looking for comments that are intended to be insulting to a person who is a part of the larger blog/forum conversation.
- We are NOT looking for insults directed to non-participants (such as celebrities, public figures etc.).
- Insults could contain profanity, racial slurs, or other offensive language. But often times, they do not.
- Comments which contain profanity or racial slurs, but are not necessarily insulting to another person are considered not insulting.
