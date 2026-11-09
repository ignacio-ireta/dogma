"""Real, offline datasets with documented quirks.

Numeric tabular sets (``load``) come from sklearn and statsmodels bundles, raw and unscaled.
Series (``load_series``) keep their NaNs. Vendored CSVs (``load_frame`` only) are copied from
``sources/`` and are deliberately not cleaned. Everything reads local files; nothing uses the
network. pandas, sklearn and statsmodels are imported inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import importlib
import io
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
from numpy.typing import NDArray

_SKLEARN = ("breast_cancer", "diabetes", "digits", "iris", "linnerud", "wine")
_STATSMODELS = ("engel", "longley", "macrodata", "spector", "stackloss")
_SERIES = ("co2", "elnino", "nile", "sunspots")
_VENDORED = ("customers", "orders", "penguins", "surveys")
_VENDORED_FILES = {name: f"{name}.csv" for name in _VENDORED}


def _pandas() -> Any:
    """Import pandas lazily; it is untyped here, so the module is typed as Any."""
    import pandas  # type: ignore[import-untyped]  # noqa: PLC0415

    return pandas


@dataclass(frozen=True)
class RealData:
    """A numeric tabular dataset: raw float64 features and an optional target.

    ``y`` is 1-D for single-target sets and 2-D (n, k) for multi-output sets (linnerud).
    ``target_name`` joins several target names with ",".
    """

    name: str
    X: NDArray[np.float64]
    y: NDArray[np.float64] | None
    feature_names: tuple[str, ...]
    target_name: str | None

    def frame(self) -> Any:
        """Return features plus target column(s) as a pandas DataFrame (a fresh object)."""
        pd = _pandas()

        table = pd.DataFrame(np.array(self.X, dtype=np.float64), columns=list(self.feature_names))
        if self.y is None:
            return table
        if self.y.ndim == 1:
            table[self.target_name or "target"] = np.array(self.y, dtype=np.float64)
            return table
        names = self.target_name.split(",") if self.target_name else []
        if len(names) != self.y.shape[1]:
            names = [f"target_{i}" for i in range(self.y.shape[1])]
        for i, column in enumerate(names):
            table[column] = np.array(self.y[:, i], dtype=np.float64)
        return table


def available() -> tuple[str, ...]:
    """Return every dataset id this module can load, sorted."""
    return tuple(sorted((*_SKLEARN, *_STATSMODELS, *_SERIES, *_VENDORED)))


def _check_name(name: object) -> str:
    if not isinstance(name, str):
        raise ValueError(f"dataset name must be a str, got {type(name).__name__}")
    if name not in available():
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(available())}")
    return name


def _float_array(values: Any) -> NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def _sklearn_real(name: str) -> RealData:
    from sklearn import datasets as skd  # noqa: PLC0415 (lazy: keeps import dogma_data numpy-only)

    if name == "iris":
        bunch = skd.load_iris()
        return RealData(
            name,
            _float_array(bunch.data),
            _float_array(bunch.target),
            tuple(bunch.feature_names),
            "target",
        )
    if name == "wine":
        bunch = skd.load_wine()
        return RealData(
            name,
            _float_array(bunch.data),
            _float_array(bunch.target),
            tuple(bunch.feature_names),
            "target",
        )
    if name == "breast_cancer":
        bunch = skd.load_breast_cancer()
        return RealData(
            name,
            _float_array(bunch.data),
            _float_array(bunch.target),
            tuple(bunch.feature_names),
            "target",
        )
    if name == "diabetes":
        bunch = skd.load_diabetes(scaled=False)
        return RealData(
            name,
            _float_array(bunch.data),
            _float_array(bunch.target),
            tuple(bunch.feature_names),
            "target",
        )
    if name == "digits":
        bunch = skd.load_digits()
        features = tuple(f"pixel_{r}_{c}" for r in range(8) for c in range(8))
        return RealData(
            name, _float_array(bunch.data), _float_array(bunch.target), features, "target"
        )
    bunch = skd.load_linnerud()
    return RealData(
        name,
        _float_array(bunch.data),
        _float_array(bunch.target),
        tuple(bunch.feature_names),
        ",".join(bunch.target_names),
    )


# (features, target) for each statsmodels set; None means "all other columns".
_STATSMODELS_SPEC: dict[str, tuple[tuple[str, ...] | None, str | None]] = {
    "longley": (None, "TOTEMP"),
    "stackloss": (("AIRFLOW", "WATERTEMP", "ACIDCONC"), "STACKLOSS"),
    "engel": (("income",), "foodexp"),
    "spector": (("GPA", "TUCE", "PSI"), "GRADE"),
    "macrodata": (None, None),
}


def _statsmodels_module(name: str) -> Any:
    return importlib.import_module(f"statsmodels.datasets.{name}")


def _statsmodels_real(name: str) -> RealData:
    frame = _statsmodels_module(name).load_pandas().data
    features_spec, target = _STATSMODELS_SPEC[name]
    if features_spec is None:
        features = tuple(c for c in frame.columns if c != target)
    else:
        features = features_spec
    y = None if target is None else _float_array(frame[target].to_numpy())
    design = _float_array(frame[list(features)].to_numpy())
    return RealData(name, design, y, features, target)


def _series_values(name: str) -> tuple[NDArray[np.float64], Any]:
    """Return (values, index) for a series id; index is a pandas index or None."""
    frame = _statsmodels_module(name).load_pandas().data
    if name == "co2":
        return _float_array(frame["co2"].to_numpy()), frame.index
    if name == "sunspots":
        return _float_array(frame["SUNACTIVITY"].to_numpy()), None
    if name == "nile":
        return _float_array(frame["volume"].to_numpy()), None
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    table = _float_array(frame[months].to_numpy())
    return np.ascontiguousarray(table).reshape(-1), None


def _check_numeric(name: str) -> None:
    if name in _SERIES or name in _VENDORED:
        raise ValueError(
            f"{name!r} is not a numeric tabular dataset; use load_series or load_frame"
        )


def load(name: str) -> RealData:
    """Load a numeric tabular dataset, raw and unscaled, as fresh arrays."""
    _check_numeric(_check_name(name))
    if name in _SKLEARN:
        return _sklearn_real(name)
    return _statsmodels_real(name)


def load_series(name: str) -> NDArray[np.float64]:
    """Load a 1-D series in time order as a fresh float64 array; NaNs are kept."""
    _check_name(name)
    if name not in _SERIES:
        raise ValueError(f"{name!r} is not a series dataset; use load or load_frame")
    values, _ = _series_values(name)
    return values


def _vendored_frame(name: str) -> Any:
    pd = _pandas()

    resource = files("dogma_data") / "data" / _VENDORED_FILES[name]
    return pd.read_csv(io.BytesIO(resource.read_bytes()))


def load_frame(name: str) -> Any:
    """Return any dataset as a pandas DataFrame (fresh each call).

    Numeric sets equal ``load(name).frame()``. Series sets are one column, keeping a date index
    when the source has one. Vendored sets are read as-is, without cleaning.
    """
    _check_name(name)
    if name in _VENDORED:
        return _vendored_frame(name)
    if name in _SERIES:
        pd = _pandas()

        values, index = _series_values(name)
        if index is None:
            index = pd.RangeIndex(len(values))
        return pd.DataFrame({name: values}, index=index)
    return load(name).frame()


def fingerprint(name: str) -> str:
    """Return the sha256 hex digest of the dataset's CSV text (index included)."""
    text = load_frame(_check_name(name)).to_csv(
        index=True, lineterminator="\n", float_format="%.17g"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def catalog() -> dict[str, dict[str, object]]:
    """Return the dataset catalog (``catalog.yaml``) keyed by dataset id."""
    import yaml  # noqa: PLC0415 (lazy: keeps import dogma_data numpy-only)

    text = (files("dogma_data") / "catalog.yaml").read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError("catalog.yaml must hold a mapping")
    result: dict[str, dict[str, object]] = {}
    for key, value in loaded.items():
        if not isinstance(value, dict):
            raise ValueError(f"catalog entry {key!r} must be a mapping")
        result[str(key)] = dict(value)
    return result
