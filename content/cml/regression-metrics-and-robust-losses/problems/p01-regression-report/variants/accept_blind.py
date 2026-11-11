import numpy as np


def regression_report(y_true, y_pred):
    t = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError("inputs must be 1-D")
    if t.shape != p.shape:
        raise ValueError("lengths differ")
    if t.size == 0:
        raise ValueError("empty")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("non-finite")
    if np.all(t == t[0]):
        raise ValueError("constant truth: R2 undefined")
    e = t - p
    mae = float(np.mean(np.abs(e)))
    mse = float(np.mean(e * e))
    d = t - t.mean()
    m = np.max(np.abs(d))
    r2 = float(1.0 - np.sum((e / m) ** 2) / np.sum((d / m) ** 2))
    return {"mae": mae, "mse": mse, "rmse": float(np.sqrt(mse)), "r2": r2}
