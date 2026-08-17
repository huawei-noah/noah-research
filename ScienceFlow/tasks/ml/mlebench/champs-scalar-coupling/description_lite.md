# Champs Scalar Coupling — Lite Task Description

## Task description
This challenge aims to predict interactions between atoms. Imaging technologies like MRI enable us to see and understand the molecular composition of tissues. Nuclear Magnetic Resonance (NMR) is a closely related technology which uses the same principles to understand the structure and dynamics of proteins and …

## Task objective
- **Input:** Each test row / notebook identified by `id` (and any features in released `test` / `public` data).
- **Output:** For each `id` in the test set, you must predict the `scalar_coupling_constant` variable.

## Target metric (evaluation)
Submissions are evaluated on the Log of the Mean Absolute Error, calculated for each scalar coupling type, and then averaged across types, so that a 1% decrease in MAE for one type provides the same improvement in score as a 1% decrease for another type. *(formula in full …

## Brief background
Hosted benchmark task; see full `description.md` for citations and organizers.

## Submission
- **File:** `submission.csv`.
- **Schema:** id,scalar_coupling_constant; one row per test key (see `sample_submission.csv`).
```
id,scalar_coupling_constant
2324604,0.0
2324605,0.0
```

## Dataset and construction
- **train.csv** - the training set, where the first column (`molecule_name`) is the name of the molecule where the coupling constant originates (the corresponding XYZ file is located at ./structures/.xyz), the second … *(see full `description.md`.)*
- **test.csv** - the test set; same info as train, without the target variable
- **sample_submission.csv** - a sample submission file in the correct format
- **structures.zip** - folder containing molecular structure (xyz) files, where the first line is the number of atoms in the molecule, followed by a blank line, and then a line for every atom, where the first column … *(see full `description.md`.)*
