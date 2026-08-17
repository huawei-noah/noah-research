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

"""Regression: pandas text dtypes (incl. numpy ``str``) must be handled before LightGBM."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype, is_numeric_dtype, is_string_dtype


def _needs_categorical_encoding(s: pd.Series) -> bool:
    """Mirror workspace solution.py logic for categorical conversion."""
    if is_numeric_dtype(s.dtype):
        return False
    if is_bool_dtype(s.dtype):
        return False
    if isinstance(s.dtype, pd.CategoricalDtype):
        return False
    if is_string_dtype(s.dtype):
        return True
    if s.dtype == object or getattr(s.dtype, "name", "") == "string":
        return True
    return not is_numeric_dtype(s.dtype) and not is_bool_dtype(s.dtype)


def _preprocess_like_solution(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if _needs_categorical_encoding(out[col]):
            out[col] = out[col].astype(str).astype("category")
    return out


def _assert_lightgbm_compatible_frame(X: pd.DataFrame) -> None:
    bad: list[tuple[str, str]] = []
    for col, dt in X.dtypes.items():
        if is_numeric_dtype(dt) or is_bool_dtype(dt):
            continue
        if isinstance(dt, pd.CategoricalDtype):
            continue
        bad.append((str(col), str(dt)))
    if bad:
        preview = ", ".join(f"{c}: {d}" for c, d in bad[:10])
        raise ValueError(f"Bad dtypes: {preview}")


def test_object_string_columns_become_category() -> None:
    raw = pd.DataFrame({"id": [1, 2], "txt": ["aa", "bb"]})
    # pandas>=2.2 may infer numpy string / StringDtype instead of object.
    assert raw["txt"].dtype == object or is_string_dtype(raw["txt"])
    X = _preprocess_like_solution(raw.drop(columns=["id"]))
    _assert_lightgbm_compatible_frame(X)
    assert str(X["txt"].dtype) == "category"


def test_pandas_nullable_string_dtype_becomes_category() -> None:
    raw = pd.DataFrame({"id": [1, 2], "txt": pd.Series(["aa", "bb"], dtype="string")})
    assert is_string_dtype(raw["txt"])
    X = _preprocess_like_solution(raw.drop(columns=["id"]))
    _assert_lightgbm_compatible_frame(X)
    assert str(X["txt"].dtype) == "category"


@pytest.mark.parametrize(
    "series_builder",
    [
        lambda: pd.Series(["x", "y"], dtype=object),
        lambda: pd.Series(pd.array(["x", "y"], dtype=pd.StringDtype())),
    ],
)
def test_common_text_dtypes_become_category(series_builder) -> None:
    s = series_builder()
    df = pd.DataFrame({"c": s})
    X = _preprocess_like_solution(df)
    _assert_lightgbm_compatible_frame(X)
    assert str(X["c"].dtype) == "category"
