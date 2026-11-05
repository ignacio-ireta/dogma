"""Cached dataset loaders.

Each loader returns a tidy :class:`pandas.DataFrame` and caches a CSV under
``<repo>/data`` on first use, so notebooks are reproducible and runnable offline
after the first download. Columns are documented in each function's docstring
and in ``DATA_SOURCES.md``.

Datasets
--------
- Wine     (sklearn, bundled)   : 13 chemical measurements, 3 cultivars.
- Diabetes (sklearn, bundled)   : 10 baseline predictors, disease-progression target.
- Penguins (seaborn, download)  : Palmer Penguins, morphology + species/island/sex.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _cache_path(name: str) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR / name


def load_wine_df() -> pd.DataFrame:
    """Wine recognition data: 178 wines x 13 chemical features + class.

    Columns: 13 numeric features (alcohol, malic_acid, ...), plus
    ``target`` (int 0/1/2) and ``target_name`` (str 'class_0'..'class_2').
    Feature scales differ wildly (e.g. proline in hundreds vs. ash near 2),
    which makes this the canonical PCA/LDA *scaling-trap* dataset.
    """
    cache = _cache_path("wine.csv")
    if cache.exists():
        return pd.read_csv(cache)
    from sklearn.datasets import load_wine

    bunch = load_wine(as_frame=True)
    df = bunch.frame.copy()
    df = df.rename(columns={"target": "target"})
    names = {i: n for i, n in enumerate(bunch.target_names)}
    df["target_name"] = df["target"].map(names)
    df.to_csv(cache, index=False)
    return df


def load_diabetes_df() -> pd.DataFrame:
    """Diabetes data: 442 patients x 10 standardized predictors + target.

    Columns: age, sex, bmi, bp, s1..s6 (already mean-centred / scaled by the
    source), plus ``target`` = a quantitative measure of disease progression
    one year after baseline. Good for multiple regression: several correlated
    blood-serum measures (s1..s6) create realistic multicollinearity.
    """
    cache = _cache_path("diabetes.csv")
    if cache.exists():
        return pd.read_csv(cache)
    from sklearn.datasets import load_diabetes

    bunch = load_diabetes(as_frame=True)
    df = bunch.frame.copy()
    df.to_csv(cache, index=False)
    return df


def load_penguins_df(dropna: bool = True) -> pd.DataFrame:
    """Palmer Penguins: 344 birds x 4 body measurements + species/island/sex.

    Columns: species, island, bill_length_mm, bill_depth_mm,
    flipper_length_mm, body_mass_g, sex. ``body_mass_g`` is ~3-4 thousand
    while ``bill_depth_mm`` is ~15-20, so this is the clustering scaling trap.

    Parameters
    ----------
    dropna : if True (default), drop rows with any missing value so downstream
        methods run without imputation surprises.
    """
    cache = _cache_path("penguins.csv")
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        import seaborn as sns

        df = sns.load_dataset("penguins")
        df.to_csv(cache, index=False)
    if dropna:
        df = df.dropna().reset_index(drop=True)
    return df


# Convenience: the four numeric morphology columns of the penguins data.
PENGUIN_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
