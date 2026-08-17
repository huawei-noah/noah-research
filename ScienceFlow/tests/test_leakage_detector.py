# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""Tests for validation-set leakage heuristics (encoder / concat false positives)."""

from scienceflow.safety.leakage_detector import validate_no_leakage


def test_label_encoder_concat_train_val_columns_not_leakage():
    code = """
import pandas as pd
from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
X_train = train_df.drop(columns=["y"])
X_val = val_df.drop(columns=["y"])
for col in ["a"]:
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_val[col]], axis=0)
    le.fit(combined.astype(str))
"""
    assert validate_no_leakage(code) == []


def test_inline_label_encoder_concat_not_leakage():
    code = """
import pandas as pd
from sklearn.preprocessing import LabelEncoder
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
X_train = train_df.drop(columns=["y"])
X_val = val_df.drop(columns=["y"])
le = LabelEncoder()
le.fit(pd.concat([X_train["a"], X_val["a"]], axis=0).astype(str))
"""
    assert validate_no_leakage(code) == []


def test_pd_concat_train_val_axis1_not_phase1():
    code = "import pandas as pd\nx = pd.concat([X_train, X_val], axis=1)\n"
    assert not any("pd.concat" in i for i in validate_no_leakage(code))


def test_model_fit_on_concat_still_flagged():
    code = """
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
train_df = pd.read_csv("dataset/train.csv")
val_df = pd.read_csv("dataset/validation.csv")
X_train = train_df[["a"]]
X_val = val_df[["a"]]
y_train = train_df["y"]
y_val = val_df["y"]
m = GradientBoostingRegressor()
m.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
"""
    issues = validate_no_leakage(code)
    assert any("AST" in i for i in issues)
    assert any("row-combining" in i for i in issues)


def test_tuple_unpack_from_preprocess_no_false_positive_on_fit():
    """Regression: val_df taint must not mark X_train tainted via multi-assign from a call."""
    code = """
import pandas as pd
from sklearn.linear_model import Ridge

def preprocess_data(train_df, val_df, test_df):
    X_train = train_df[["a"]]
    X_val = val_df[["a"]]
    y_train = train_df["y"]
    y_val = val_df["y"]
    return X_train, X_val, None, y_train, y_val, None, None, None

def main():
    train_df = pd.read_csv("dataset/train.csv")
    val_df = pd.read_csv("dataset/validation.csv")
    test_df = pd.read_csv("dataset/test.csv")
    X_train, X_val, X_test, y_train, y_val, a, b, c = preprocess_data(
        train_df, val_df, test_df
    )
    m = Ridge()
    m.fit(X_train, y_train)

main()
"""
    assert validate_no_leakage(code) == []
