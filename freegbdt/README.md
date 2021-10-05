# FreeGBDT

Welcome to the repository for the paper "Enhancing Transformers with Gradient Boosted Decision Trees for NLI Fine-Tuning" published in Findings of ACL 2021 https://aclanthology.org/2021.findings-acl.26/.

## Getting started

Download all the necessary data:

```
./prepare_data.sh
```

This will create a `data/` directory containing the CB, RTE, CNLI, ANLI and QNLI datasets as used in the paper.

Install the requirements:

```
pip install -r requirements.txt
```

Experiments were carried out with Python version 3.7.7 and the following hardware:

- GPU: Tesla V100-SXM2-16GB
- CPU: Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz

## How to use

__Terminology__

- `L`: the amount of classes, 3 for classification between `entailment`, `neutral`, `contradiction`. 1 for binary classification between `entailment`, `not_entailment`.
- `C`: the amount of features i. e. 1024 in RoBERTa-large.
- `E`: epochs the model is trained for (10 by default).

### Neural Network

To train a neural network from the paper:

```
python train_base_model.py --task="<task>" --model_path="<path/to/roberta-large-mnli>"
```

This will store output in a folder in the "logs/" folder (relative to the script file) e. g. `logs/CB_02091810_c0ce57`. This will contain the files:

- `params.json` - hyperparamaters of the neural network and metadata.
- `out.log` - stdout and stderr of the run.
- `features.joblib` - features collected during training (the FreeGBDT training data) and features extracted after training (the standard GBDT training data). A dictionary with the keys:
    - `"train_features"`: numpy array, `E x |train| x C`. Features collected during training for the FreeGBDT.
    - `train_labels`: numpy array, `E x |train| x L`. The labels corresponding to the train features.
    - `"extracted_train_features"`: numpy array, `(E + 1) x |train| x C`. train features extracted by one additional forward pass at the start of fine-tuning and after each epoch. Always in the order of the original train set.
    - `"val_features"`: numpy array, `(E + 1) x |dev| x C`. Same as `"extracted_train_features"`, but for dev data. Always in the order of the original dev set.
    - `"test_features"`: numpy array, `|test| x C`. Test features, extracted once after fine-tuning.

Load it via `joblib`:

```python
import joblib

x = joblib.load("/path/to/result.joblib")
```

### The GBDTs

To train a GBDT from the paper:

```
python train_tree_model.py --base="<path/to/base/logdir>"
```

where the path given to base is e. g. `logs/CB_02091810_c0ce57`.
This will add the file `result.joblib` to the log directory. This file contains a dictionary with the keys:
- `"tree_scores"`, `"regular_tree_scores"`, `"nn_scores"`. List of numpy arrays, `E x 5`, `(E + 1) x 5` and `(E + 1) x 5`. Contain the scores (accuracy) for the FreeGBDT, standard GBDT and neural network, respectively on the dev set. The last dimension contains scores for trees trained with 1, 10, 20, 30 and 40 boosting rounds (we train each tree for each dataset to reduce complexity).
- `"test_tree_preds"`, `"test_regular_preds"`, `"test_nn_preds"`. List of numpy arrays, `5 x |test| x L`. Predictions for the test set. The first dimension differentiates between trees trained with 1, 10, 20, 30 and 40 boosting rounds.

## Reproducing Results from the paper

You can pass a seed to `train_base_model.py` via the `--seed` flag:

```
python train_base_model.py --seed="<seed>" --task="<task>" --model_path="<path/to/roberta-large-mnli>"
```

We have done our best to make runs deterministic, but we can't guarantee that the seed will produce our exact score for all current and future hardware.

### Results on the development sets (Table 2)

- Seeds used for CB: `31098,97725,81443,99815,26703,47373,10417,21086,63919,7786,58155,76683,3385,7656,26317,69794,92948,27902,29839,5698`
- Seeds used for RTE: `79477,98417,56652,82120,49049,66940,41084,4582,92043,86420,90814,80201,87412,19879,80139,90724,87598,34060,23026,78613`
- Seeds used for CNLI: `50404,14011,84602,23641,24869,53066,48084,24400,97431,26236,54993,4107,61203,275,93824,42847,70868,35106,61461,10882`
- Seeds used for ANLI: `57557,86102,80305,63004,52326,43803,3769,68785,92816,61047,1677,38798,60800,16657,88336,18926,65152,11440,89429,80630`
- Seeds used for QNLI: `54090,22080,22021,64498,90882,77197,18625,43851,11370,12090,27103,91054,90129,52914,18172,5317,43829,10968,2865,68635`

### Results on the test sets (Table 4 and Table 5)

- Seed used for CB: `7786`
- Seed used for RTE: `80139`
- Seed used for CNLI: `61461`
- Seed used for ANLI: `11440`
- Seed used for QNLI: `43851`
