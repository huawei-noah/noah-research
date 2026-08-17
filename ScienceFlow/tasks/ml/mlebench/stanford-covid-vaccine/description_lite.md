# Stanford Covid Vaccine — Lite Task Description

## Task description
In this competition, we are looking to leverage the data science expertise of the Kaggle community to develop models and design rules for RNA degradation. Your model will predict likely degradation rates at each base of an RNA molecule, trained on a subset of an Eterna dataset comprising over 3000 RNA molecules (which …

## Task objective
- **Input:** Each test sample as defined by the competition `test` split and `sample_submission.csv` rows.
- **Output:** For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row.

## Target metric (evaluation)
Submissions are scored using MCRMSE, mean columnwise root mean squared error: where is the number of scored ground truth target columns, and and are the actual and predicted values, respectively. From the `Data` page: There are multiple ground truth values provided in the …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id_seqpos, … (remaining columns per sample_submission.csv); one row per test key (see `sample_submission.csv`).
```
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
```

## Dataset and construction
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format
- `id` - An arbitrary identifier for each sample.
