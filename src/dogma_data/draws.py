"""Seeded draws from the real datasets: random rows and columns, and windows of a series.

A test that used to build its bulk input with ``rng.normal(size=(n, d))`` asks for the same
shape here and gets real values: real scales, correlated columns, skew, and the ties that come
from measurements rounded to a few decimals. The draw is decided by the generator alone, so a
key from :func:`dogma_data.rng` makes it repeatable and lets the audit salt vary it.

The datasets are read once per process and kept read-only; every call returns fresh arrays.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from numpy.typing import NDArray

from dogma_data import real

# HuggingFace tables that are fully numeric: (feature columns, target column).
_NUMERIC_TABLES: dict[str, tuple[tuple[str, ...], str]] = {
    "california_housing": (
        (
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ),
        "MedHouseVal",
    ),
}
_SERIES = ("co2", "elnino", "nile", "sunspots")


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values.setflags(write=False)
    return values


@cache
def _table(name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if name in _NUMERIC_TABLES:
        features, target = _NUMERIC_TABLES[name]
        frame = real.load_frame(name)
        design = np.array(frame[list(features)].to_numpy(), dtype=np.float64)
        return _frozen(design), _frozen(np.array(frame[target].to_numpy(), dtype=np.float64))
    data = real.load(name)
    if data.y is None:
        return _frozen(data.X), _frozen(np.empty((data.X.shape[0], 0), dtype=np.float64))
    return _frozen(data.X), _frozen(data.y)


@cache
def _series(name: str) -> NDArray[np.float64]:
    return _frozen(real.load_series(name))


def sampleable() -> tuple[str, ...]:
    """Return the dataset ids :func:`sample_rows` accepts, sorted."""
    names = [name for name in real.available() if _is_numeric(name)]
    return tuple(sorted(names))


def _is_numeric(name: str) -> bool:
    if name in _NUMERIC_TABLES:
        return True
    try:
        real._check_numeric(name)
    except ValueError:
        return False
    return True


def _count(value: object, label: str, limit: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise ValueError(f"{label} must be an int, got {type(value).__name__}")
    if not 1 <= int(value) <= limit:
        raise ValueError(f"{label} must be between 1 and {limit} for {name!r}, got {int(value)}")
    return int(value)


def sample_rows(
    name: str,
    gen: np.random.Generator,
    n: int,
    d: int | None = None,
    *,
    standardize: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw ``n`` distinct rows and ``d`` distinct columns of a numeric dataset.

    Returns ``(X, y)``. ``X`` is a fresh float64 array of shape ``(n, d)``; ``d=None`` keeps
    every column in the dataset's own order. ``y`` holds the target of the drawn rows: 1-D, or
    2-D ``(n, k)`` for a multi-output set, or ``(n, 0)`` for a set with no target. Rows and
    columns are drawn without replacement and come back in drawn order, so real duplicates and
    ties stay but no row is repeated by the draw.

    ``standardize=True`` centres and scales each column with the mean and the population
    standard deviation of the *whole* dataset, so a small draw keeps its real spread. A column
    that is constant in the dataset becomes zeros.
    """
    if not isinstance(name, str) or not _is_numeric(real._check_name(name)):
        raise ValueError(f"{name!r} has no numeric table; use one of: {', '.join(sampleable())}")
    design, target = _table(name)
    rows = _count(n, "n", design.shape[0], name)
    picked_rows = gen.choice(design.shape[0], size=rows, replace=False)
    if d is None:
        picked_cols = np.arange(design.shape[1])
    else:
        cols = _count(d, "d", design.shape[1], name)
        picked_cols = gen.choice(design.shape[1], size=cols, replace=False)
    out = np.array(design[np.ix_(picked_rows, picked_cols)], dtype=np.float64)
    if standardize:
        mean = design.mean(axis=0)[picked_cols]
        std = design.std(axis=0)[picked_cols]
        out = (out - mean) / np.where(std > 0.0, std, 1.0)
    return out, np.array(target[picked_rows], dtype=np.float64)


def sample_window(
    name: str, gen: np.random.Generator, n: int, *, keep_nan: bool = False
) -> NDArray[np.float64]:
    """Draw ``n`` consecutive values of a series, in time order, as a fresh float64 array.

    The start is uniform over every position that leaves room for ``n`` values. Missing values
    are dropped before the window is taken unless ``keep_nan=True``.
    """
    if not isinstance(name, str) or name not in _SERIES:
        raise ValueError(f"{name!r} is not a series; series: {', '.join(_SERIES)}")
    values = _series(name)
    if not keep_nan:
        values = values[~np.isnan(values)]
    length = _count(n, "n", values.shape[0], name)
    start = int(gen.integers(0, values.shape[0] - length + 1))
    return np.array(values[start : start + length], dtype=np.float64)
