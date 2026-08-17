---
name: tabular_preprocess_speedup
category: data_processing
tags: [tabular, performance, preprocessing, encoding, vectorized]
task_types: [Tabular]
phase: [drafting, improving]
priority: 88
has_post_process: false
---

# Tabular Preprocessing — Fast Patterns

Write preprocessing code that avoids Python-level loops over rows or columns.
This matters especially with **wide tables (100+ columns)** and/or mixed-type data.

---

## 1. Categorical Encoding — NEVER use LabelEncoder + apply

**Forbidden pattern** (pure-Python per-row loop, extremely slow on wide tables):

```python
# SLOW — do NOT write this
for col in cat_cols:
    le = LabelEncoder()
    le.fit(train[col].unique())
    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
```

### Preferred: LightGBM / CatBoost native categorical (zero extra encoding)

```python
import lightgbm as lgb

cat_features = [col for col in feature_cols if df[col].dtype == "object"]
train_data = lgb.Dataset(
    X_train, label=y_train,
    feature_name=feature_cols,
    categorical_feature=cat_features,  # LightGBM handles internally
)
```

Pass raw string columns directly; LightGBM encodes internally with no Python loop.

### Fallback: vectorized factorize (when numeric codes are needed)

```python
# Fit on train, apply to val/test using the same mapping
def encode_categoricals(train_df, other_dfs, cat_cols):
    mappings = {}
    for col in cat_cols:
        train_df[col] = train_df[col].astype(str).fillna("__NA__")
        codes, uniques = pd.factorize(train_df[col])
        mapping = {v: i for i, v in enumerate(uniques)}
        mappings[col] = mapping
        train_df[col] = codes
    for df in other_dfs:
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("__NA__").map(mappings[col]).fillna(-1).astype(int)
    return train_df, other_dfs
```

### Alternative: OrdinalEncoder (sklearn, batch fit_transform)

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype="float32")
train[cat_cols] = enc.fit_transform(train[cat_cols].astype(str).fillna("__NA__"))
val[cat_cols]   = enc.transform(val[cat_cols].astype(str).fillna("__NA__"))
test[cat_cols]  = enc.transform(test[cat_cols].astype(str).fillna("__NA__"))
```

---

## 2. Numeric Columns — batch operations, no per-column loop

**Slow pattern:**

```python
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(train[col].median())
    df[col] = df[col].clip(q01[col], q99[col])
```

**Fast equivalent:**

```python
# 1. Coerce types in one pass
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

# 2. Compute stats once on train, fill all DataFrames at once
medians = train[num_cols].median()           # Series
q01     = train[num_cols].quantile(0.01)
q99     = train[num_cols].quantile(0.99)

for df in [train, val, test]:
    df[num_cols] = df[num_cols].fillna(medians)
    df[num_cols] = df[num_cols].clip(lower=q01, upper=q99, axis=1)
```

---

## 3. IO — avoid repeated read_csv, use parquet cache

Wide CSVs with many mixed-type columns are slow to parse on every run.
Cache after the first load:

```python
from pathlib import Path
import pandas as pd

CACHE = Path("dataset/.preprocess_cache.parquet")

def load_or_cache(csv_path: str) -> pd.DataFrame:
    cache = Path(csv_path).with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(csv_path, low_memory=False)
    df.to_parquet(cache, index=False)
    return df
```

Also pre-declare `dtype` for known columns to skip inference:

```python
train = pd.read_csv("train.csv", dtype={"id": str, "label": float}, low_memory=False)
```

---

## 4. Self-check before running

- [ ] No `LabelEncoder` + `.apply(lambda x: le.transform([x])...)` in solution
- [ ] No `for col in cat_cols: df[col] = df[col].apply(...)` (per-row Python loop)
- [ ] Numeric `fillna` / `clip` done at DataFrame level, not in a Python for-loop
- [ ] `pd.to_numeric` called once per DataFrame, not per column
- [ ] Stats (median, quantiles) computed on **train split only**, applied to val/test
- [ ] LightGBM / CatBoost `categorical_feature` used when possible (avoids all encoding)
- [ ] parquet cache used if solution.py will be re-run during tuning
