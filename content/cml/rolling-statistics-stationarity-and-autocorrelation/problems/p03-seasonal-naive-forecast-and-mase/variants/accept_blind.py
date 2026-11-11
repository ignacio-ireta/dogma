from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _vec(a, name: str) -> NDArray[np.float64]:
    try:
        arr = np.array(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite 1-D sequence")
    return arr


def seasonal_naive_forecast_and_mase(
    train: Sequence[float] | NDArray[np.float64],
    test: Sequence[float] | NDArray[np.float64],
    season: int,
) -> tuple[NDArray[np.float64], float]:
    y = _vec(train, "train")
    t = _vec(test, "test")
    if t.size == 0:
        raise ValueError("test is empty")
    if (
        isinstance(season, (bool, np.bool_))
        or not isinstance(season, (int, np.integer))
        or season < 1
    ):
        raise ValueError("season must be a positive integer")
    season = int(season)
    n = y.size
    if n < season + 1:
        raise ValueError("train too short")
    H = t.size
    idx = np.arange(1, H + 1)
    pos = n + idx - season * (-(-idx // season))  # 1-based
    forecast = y[pos - 1].astype(np.float64)
    with np.errstate(over="ignore"):
        raw = np.abs(y[season:] - y[:-season])
    if not np.any(raw != 0):
        raise ValueError("scale is zero")
    big = max(np.max(np.abs(y)), np.max(np.abs(t)))
    ys, ts, fs = y / big, t / big, forecast / big
    s = np.mean(np.abs(ys[season:] - ys[:-season]))
    if s == 0:
        raise ValueError("scale is zero")
    mase = float(np.mean(np.abs(ts - fs)) / s)
    return forecast, mase
