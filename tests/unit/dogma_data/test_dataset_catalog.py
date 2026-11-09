"""Real datasets: offline loading, catalog agreement, documented quirks, vendored copies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import dogma_data
from dogma_data import real

REPO = Path(__file__).resolve().parents[3]
SOURCES = {
    "penguins.csv": REPO / "sources/mvs/data/penguins.csv",
    "customers.csv": REPO / "sources/interview-prep/data/customers.csv",
    "orders.csv": REPO / "sources/interview-prep/data/orders.csv",
    "surveys.csv": REPO / "sources/interview-prep/data/surveys.csv",
}
CATALOG = real.catalog()


def _vendored_path(filename: str) -> Path:
    return Path(str(dogma_data.__file__)).parent / "data" / filename


def _abs_corr_pairs(design: NDArray[np.float64], threshold: float) -> int:
    corr = np.corrcoef(design, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)
    return int(np.sum(upper > threshold))


def _skew(values: NDArray[np.float64]) -> float:
    centred = values - values.mean()
    return float(np.mean(centred**3) / np.std(values) ** 3)


def test_available_matches_catalog() -> None:
    assert dogma_data.available() == tuple(sorted(CATALOG))


@pytest.mark.parametrize("name", dogma_data.available())
def test_frame_shape_matches_catalog(name: str) -> None:
    frame = dogma_data.load_frame(name)
    assert [int(frame.shape[0]), int(frame.shape[1])] == CATALOG[name]["shape"], name


@pytest.mark.parametrize("name", dogma_data.available())
def test_fingerprint_matches_catalog(name: str) -> None:
    got = dogma_data.fingerprint(name)
    assert got == CATALOG[name]["sha256"], (
        f"dataset {name!r} changed: the library or uv.lock moved, so its CSV text differs from "
        "the catalog. Re-check the problems that use it (grep -rl dogma_data content/), then "
        "update catalog.yaml and bump those problems' versions."
    )


# --- quirks: the facts the content tests rely on (measured, bounds are generous) ---


def test_iris_has_one_duplicate_row_and_one_decimal_values() -> None:
    iris = real.load("iris")
    assert len(iris.X) - len(np.unique(iris.X, axis=0)) == 1
    np.testing.assert_allclose(np.round(iris.X, 1), iris.X)


def test_wine_proline_dominates_unscaled_variance() -> None:
    wine = real.load("wine")
    variances = wine.X.var(axis=0)
    proline = wine.feature_names.index("proline")
    assert variances[proline] / variances.sum() > 0.99


def test_breast_cancer_scales_and_correlation() -> None:
    bc = real.load("breast_cancer")
    means = bc.X.mean(axis=0)
    assert means.min() < 0.01
    assert means.max() > 800
    assert _abs_corr_pairs(bc.X, 0.9) >= 20


def test_breast_cancer_is_nearly_separable_when_standardised() -> None:
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    bc = real.load("breast_cancer")
    standardised = (bc.X - bc.X.mean(axis=0)) / bc.X.std(axis=0)
    assert bc.y is not None
    clf = LogisticRegression(C=1e4, max_iter=20000).fit(standardised, bc.y)
    assert clf.score(standardised, bc.y) > 0.99


def test_diabetes_is_raw_with_binary_sex() -> None:
    diabetes = real.load("diabetes")
    sex = diabetes.X[:, diabetes.feature_names.index("sex")]
    assert set(np.unique(sex).tolist()) == {1.0, 2.0}
    assert float(np.abs(diabetes.X).max()) > 0.5  # raw values, not standardised to ~0.05


def test_digits_has_zero_variance_and_zero_iqr_columns() -> None:
    digits = real.load("digits")
    assert int(np.sum(digits.X.std(axis=0) == 0)) == 3
    iqr = np.percentile(digits.X, 75, axis=0) - np.percentile(digits.X, 25, axis=0)
    assert int(np.sum(iqr == 0)) >= 10


def test_linnerud_is_multi_output() -> None:
    linnerud = real.load("linnerud")
    assert linnerud.y is not None
    assert linnerud.y.ndim == 2
    assert linnerud.y.shape == (20, 3)
    assert linnerud.target_name == "Weight,Waist,Pulse"


def test_longley_design_is_ill_conditioned() -> None:
    longley = real.load("longley")
    design = np.column_stack([np.ones(len(longley.X)), longley.X])
    cond = float(np.linalg.cond(design))
    assert 1e9 < cond < 1e10


def test_stackloss_observation_21_is_an_outlier() -> None:
    stackloss = real.load("stackloss")
    assert stackloss.y is not None
    design = np.column_stack([np.ones(len(stackloss.X)), stackloss.X])
    beta = np.linalg.lstsq(design, stackloss.y, rcond=None)[0]
    resid = stackloss.y - design @ beta
    hat = np.diag(design @ np.linalg.pinv(design.T @ design) @ design.T)
    sigma = np.sqrt(np.sum(resid**2) / (len(resid) - design.shape[1]))
    studentised = resid / (sigma * np.sqrt(1 - hat))
    assert int(np.argmax(np.abs(studentised))) == 20
    assert abs(studentised[20]) > 2.0


def test_engel_is_right_skewed() -> None:
    engel = real.load("engel")
    assert engel.y is not None
    assert _skew(engel.X[:, 0]) > 2.0
    assert _skew(engel.y) > 1.0


def test_spector_grade_is_binary() -> None:
    spector = real.load("spector")
    assert spector.y is not None
    assert set(np.unique(spector.y).tolist()) == {0.0, 1.0}


def test_co2_keeps_59_nans_and_a_weekly_index() -> None:
    co2 = dogma_data.load_series("co2")
    assert co2.shape == (2284,)
    assert int(np.isnan(co2).sum()) == 59
    assert dogma_data.load_frame("co2").index.dtype.kind == "M"


def test_sunspots_acf_peaks_near_eleven_years() -> None:
    sunspots = dogma_data.load_series("sunspots")
    centred = sunspots - sunspots.mean()
    denom = float(np.sum(centred * centred))
    acf = {k: float(np.sum(centred[:-k] * centred[k:]) / denom) for k in range(6, 17)}
    assert max(acf, key=lambda k: acf[k]) in (10, 11)


def test_nile_has_a_level_change_near_1898() -> None:
    nile = dogma_data.load_series("nile")
    best = max(range(20, 80), key=lambda k: abs(nile[:k].mean() - nile[k:].mean()))
    assert 1870 + best in range(1897, 1900)
    assert nile[:best].mean() - nile[best:].mean() > 150


def test_elnino_is_monthly_and_seasonal() -> None:
    elnino = dogma_data.load_series("elnino")
    assert elnino.shape == (61 * 12,)
    assert int(np.isnan(elnino).sum()) == 0
    centred = elnino - elnino.mean()
    lag12 = float(np.corrcoef(centred[:-12], centred[12:])[0, 1])
    assert lag12 > 0.5


def test_penguins_has_missing_values_and_string_categoricals() -> None:
    penguins = dogma_data.load_frame("penguins")
    assert int(penguins["sex"].isna().sum()) == 11
    assert int(penguins["bill_length_mm"].isna().sum()) == 2
    assert penguins["species"].dtype.kind in "OU" or str(penguins["species"].dtype) in (
        "str",
        "string",
        "object",
    )


def test_customers_is_messy_on_purpose() -> None:
    customers = dogma_data.load_frame("customers")
    assert int(customers["customer_id"].duplicated().sum()) == 50
    ages = customers["age"]
    assert int(((ages <= 0) | (ages >= 150)).sum()) == 15
    assert int(customers["gender"].isin([" ", "m", "f"]).sum()) > 0


def test_orders_and_surveys_have_missing_values() -> None:
    orders = dogma_data.load_frame("orders")
    surveys = dogma_data.load_frame("surveys")
    assert int(orders["amount"].isna().sum()) == 300
    assert int(surveys["satisfaction_score"].isna().sum()) == 145
    assert int(surveys["nps_score"].isna().sum()) == 100


# --- HF snapshots (vendored by scripts/sync_hf_datasets.py) ---


def test_adult_keeps_question_mark_missing_markers_and_imbalance() -> None:
    adult = dogma_data.load_frame("adult")
    marks = {col: int((adult[col] == "?").sum()) for col in adult.columns}
    assert marks["workclass"] == 338
    assert marks["occupation"] == 340
    assert marks["native.country"] == 98
    assert abs(float((adult["income"] == ">50K").mean()) - 0.1878) < 1e-3


def test_california_target_is_censored_and_features_heavy_tailed() -> None:
    houses = dogma_data.load_frame("california_housing")
    assert float(houses["MedHouseVal"].max()) == 5.00001
    assert int((houses["MedHouseVal"] == 5.00001).sum()) == 241
    occupancy = houses["AveOccup"]
    assert float(occupancy.max()) > 50 * float(occupancy.median())


def test_gsm8k_answers_end_with_the_final_number() -> None:
    answers = dogma_data.load_frame("gsm8k")["answer"]
    assert bool(answers.str.contains(r"#### -?[\d,.]+\s*$", regex=True).all())


@pytest.mark.parametrize(
    ("name", "opening"),
    [
        ("gutenberg_frankenstein", "You will rejoice to hear that no disaster"),
        ("gutenberg_pride_prejudice", "IT is a truth universally acknowledged"),
        ("gutenberg_walden", "When I wrote the following pages"),
    ],
)
def test_text_starts_at_the_first_sentence_and_is_real_prose(name: str, opening: str) -> None:
    text = dogma_data.load_text(name)
    assert text.startswith(opening)
    assert text.endswith("\n")
    words = text.lower().split()
    counts = sorted(
        (words.count(w) for w in set(words)), reverse=True
    )  # Zipf-like: a few very common words, a long tail of rare ones
    assert counts[0] > 50 * counts[len(counts) // 2]


def test_load_text_rejects_tables() -> None:
    with pytest.raises(ValueError):
        dogma_data.load_text("adult")


# --- vendored copies and the load contract ---


@pytest.mark.parametrize("filename", sorted(SOURCES))
def test_vendored_file_is_byte_identical_to_source(filename: str) -> None:
    assert _vendored_path(filename).read_bytes() == SOURCES[filename].read_bytes()


def test_load_returns_fresh_arrays() -> None:
    first = real.load("iris")
    first.X[:] = 0.0
    assert first.y is not None
    first.y[:] = -1.0
    second = real.load("iris")
    assert float(second.X.sum()) > 0
    assert second.y is not None
    assert set(np.unique(second.y).tolist()) == {0.0, 1.0, 2.0}


def test_series_returns_fresh_arrays() -> None:
    first = dogma_data.load_series("nile")
    first[:] = 0.0
    assert float(dogma_data.load_series("nile").sum()) > 0


def test_frame_equals_load_frame_for_numeric_sets() -> None:
    for name in ("iris", "longley", "linnerud"):
        expected = real.load(name).frame()
        assert list(dogma_data.load_frame(name).columns) == list(expected.columns)
        np.testing.assert_array_equal(dogma_data.load_frame(name).to_numpy(), expected.to_numpy())


@pytest.mark.parametrize(
    ("call", "bad"),
    [
        (real.load, "nope"),
        (real.load, "penguins"),
        (real.load, 3),
        (dogma_data.load_series, "iris"),
        (dogma_data.load_frame, "nope"),
        (real.load, "adult"),
        (real.load, "gutenberg_walden"),
        (dogma_data.load_series, "adult"),
        (dogma_data.fingerprint, "nope"),
    ],
)
def test_unknown_or_wrong_kind_raises_value_error(call: object, bad: object) -> None:
    with pytest.raises(ValueError):
        call(bad)  # type: ignore[operator]


def test_unknown_name_error_lists_available_names() -> None:
    with pytest.raises(ValueError, match="iris"):
        real.load("nope")
