# Us Patent Phrase To Phrase Matching — Lite Task Description

## Task description
Can you extract meaning from a large, text-based dataset derived from inventions? Here's your chance to do so. The U.S. Patent and Trademark Office (USPTO) offers one of the largest repositories of scientific, technical, and commercial information in the world through its Open Data Portal. Patents are a form of …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`.

## Target metric (evaluation)
Submissions are evaluated on the Pearson correlation coefficient between the predicted and actual similarity `score`s.

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv` — Submit via Kaggle **Notebook / Code competition**.
- **Schema:** id,score; one row per test key (see `sample_submission.csv`).
```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
```

## Dataset and construction
- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
