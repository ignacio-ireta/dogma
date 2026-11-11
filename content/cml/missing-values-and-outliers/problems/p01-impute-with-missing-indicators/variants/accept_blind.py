import numpy as np
import pandas as pd


def impute_with_indicators(df, numeric, strategy="median"):
    if strategy not in ("median", "mean"):
        raise ValueError("bad strategy")
    if len(set(numeric)) != len(numeric):
        raise ValueError("duplicate name")
    for name in numeric:
        if name not in df.columns:
            raise ValueError("missing column")
        col = df[name]
        if not pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            raise ValueError("not numeric")
    gapped = [n for n in numeric if df[n].isna().any()]
    for name in gapped:
        if f"{name}_was_missing" in df.columns:
            raise ValueError("indicator exists")
    out = df.copy()
    for name in gapped:
        observed = df[name].dropna().astype(float)
        if observed.empty:
            raise ValueError("no observed value")
        fill = observed.median() if strategy == "median" else observed.mean()
        out[name] = df[name].astype(float).fillna(float(fill))
    for name in gapped:
        out[f"{name}_was_missing"] = df[name].isna().astype(np.int8)
    return out
