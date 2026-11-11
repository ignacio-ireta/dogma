from collections import Counter

import numpy as np
import pandas as pd


def fit_transform_pipeline(train, test, numeric, categorical):  # noqa: PLR0912
    listed = list(numeric) + list(categorical)
    if not listed:
        raise ValueError("no columns")
    if len(set(listed)) != len(listed):
        raise ValueError("duplicate")
    for frame in (train, test):
        for c in listed:
            if c not in frame.columns:
                raise ValueError("missing column")
    tr_blocks, te_blocks = [], []
    for c in numeric:
        tr = train[c].to_numpy(dtype=float)
        te = test[c].to_numpy(dtype=float)
        obs = tr[~np.isnan(tr)]
        if obs.size == 0:
            raise ValueError("no observed numeric value")
        med = float(np.median(obs))
        tr = np.where(np.isnan(tr), med, tr)
        te = np.where(np.isnan(te), med, te)
        if tr.max() == tr.min():
            mean, scale = float(tr[0]), 1.0
        else:
            mean = float(tr.mean())
            c = tr - mean  # noqa: PLW2901
            peak = float(np.max(np.abs(c)))
            scale = peak * float(np.sqrt(np.mean((c / peak) ** 2)))
        tr_blocks.append(((tr - mean) / scale).reshape(-1, 1))
        te_blocks.append(((te - mean) / scale).reshape(-1, 1))
    for c in categorical:
        vals = train[c].tolist()
        obs = [v for v in vals if not pd.isna(v)]
        if not obs:
            raise ValueError("no observed categorical value")
        cnt = Counter(obs)
        top = max(cnt.values())
        mode = min(v for v, k in cnt.items() if k == top)
        filled = [mode if pd.isna(v) else v for v in vals]
        cats = sorted(set(filled))
        pos = {v: i for i, v in enumerate(cats)}
        trb = np.zeros((len(train), len(cats)))
        for r, v in enumerate(filled):
            trb[r, pos[v]] = 1.0
        teb = np.zeros((len(test), len(cats)))
        for r, v in enumerate(test[c].tolist()):
            v = mode if pd.isna(v) else v  # noqa: PLW2901
            if v in pos:
                teb[r, pos[v]] = 1.0
        tr_blocks.append(trb)
        te_blocks.append(teb)
    return np.hstack(tr_blocks).astype(float), np.hstack(te_blocks).astype(float)
