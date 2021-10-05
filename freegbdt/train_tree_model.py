# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import json
import numpy as np
from joblib import dump, load
import lightgbm as lgb
from tqdm.auto import tqdm
import argparse
from pathlib import Path
import datasets


def to_predictions(preds):
    if len(preds.shape) == 1:
        preds = preds[:, np.newaxis]

    if n_labels == 1:
        preds = preds[:, 0] > 0.5
    else:
        preds = np.argmax(preds, 1)

    return preds


def train(
    train_features,
    train_labels,
    valid_features,
    valid_labels,
    round_sweep,
    test_features,
):
    all_preds = []
    all_test_preds = []

    dtrain = lgb.Dataset(train_features, label=train_labels)

    for n in round_sweep:
        eval_results = {}
        lgb_model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=n,
            verbose_eval=20,
            evals_result=eval_results,
        )

        preds = lgb_model.predict(valid_features)
        if test_features is not None:
            all_test_preds.append(to_predictions(lgb_model.predict(test_features)))
        all_preds.append(preds)

    scores = []

    for x in all_preds:
        score = (to_predictions(x) == val_data["label"].values).mean()
        scores.append(score)

    return np.array(scores), np.array(all_test_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, required=True)

    args = parser.parse_args()
    args.base = Path(args.base)

    params = json.load(open(args.base / "params.json"))
    hidden_dim = int(params["hidden_dim"])

    dataset = datasets.DATASETS[params["task"].strip("'")]
    train_data, val_data, test_data = dataset.load()

    features = load(args.base / "features.joblib")
    train_features = np.stack(features["train_features"], 0)
    extracted_train_features = np.stack(features["extracted_train_features"], 0)
    train_labels = np.stack(features["train_labels"], 0)
    val_features = np.stack(features["val_features"], 0)
    test_features = features["test_features"]

    n_labels = dataset.get_n_labels()

    lgb_params = {
        "objective": "binary" if n_labels == 1 else "multiclass",
        "metric": "binary_error" if n_labels == 1 else "multi_error",
        "num_class": n_labels,
        "num_leaves": 256,
        "max_depth": -1,
        "learning_rate": 0.1,
        "verbosity": -1,
        "seed": 1234,
    }

    tree_scores = []

    for epoch in tqdm(range(len(train_features))):
        score, test_tree_preds = train(
            train_features[: (epoch + 1)].reshape((-1, hidden_dim + n_labels))[
                :, 0:-n_labels
            ],
            train_labels[: (epoch + 1)].reshape((-1)),
            val_features[epoch + 1][:, :-n_labels],
            val_data["label"].values,
            round_sweep=[1, 10, 20, 30, 40],
            test_features=test_features[:, :-n_labels],
        )

        print(f"Tree {epoch}: {score}")

        tree_scores.append(score)

    regular_scores = []
    nn_scores = []

    for epoch in tqdm(range(len(extracted_train_features))):
        regular_score, test_regular_preds = train(
            extracted_train_features[epoch][:, :-n_labels],
            train_data["label"].values,
            val_features[epoch][:, :-n_labels],
            val_data["label"].values,
            round_sweep=[1, 10, 20, 30, 40],
            test_features=test_features[:, :-n_labels],
        )

        nn_preds = to_predictions(val_features[epoch][:, -n_labels:])
        nn_score = (nn_preds == val_data["label"].values).mean()

        print(f"NN {epoch}: {nn_score}")
        print(f"Regular tree {epoch}: {regular_score}")

        nn_scores.append(nn_score)
        regular_scores.append(regular_score)

        if epoch == len(extracted_train_features) - 1:
            test_nn_preds = to_predictions(test_features[:, -n_labels:])

    dump(
        {
            "tree_scores": tree_scores,
            "regular_tree_scores": regular_scores,
            "nn_scores": nn_scores,
            "test_nn_preds": test_nn_preds,
            "test_tree_preds": test_tree_preds,
            "test_regular_preds": test_regular_preds,
        },
        args.base / "result.joblib",
    )
