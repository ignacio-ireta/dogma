"""Seeded draws from the real datasets: shapes, determinism, real values, validation."""

from __future__ import annotations

import numpy as np
import pytest

import dogma_data
from dogma_data import load, load_series, rng, sample_rows, sample_window, sampleable


def test_sampleable_lists_every_numeric_table() -> None:
    names = sampleable()
    assert "california_housing" in names
    assert "wine" in names
    assert "co2" not in names
    assert "penguins" not in names
    assert set(names) <= set(dogma_data.available())


@pytest.mark.parametrize("name", sampleable())
def test_rows_are_real_rows_of_the_dataset(name: str) -> None:
    drawn, y = sample_rows(name, rng(f"draws:{name}"), 5)
    if name == "california_housing":
        frame = dogma_data.load_frame(name)
        full = frame.drop(columns="MedHouseVal").to_numpy(dtype=np.float64)
        target = frame["MedHouseVal"].to_numpy(dtype=np.float64)
    else:
        data = load(name)
        full, target = data.X, data.y
    assert drawn.shape == (5, full.shape[1])
    assert drawn.dtype == np.float64
    for i, row in enumerate(drawn):
        hits = np.flatnonzero((full == row).all(axis=1))
        assert hits.size >= 1, "a drawn row is not in the dataset"
        if target is not None:
            assert any(np.array_equal(target[hit], y[i]) for hit in hits)
    if target is None:
        assert y.shape == (5, 0), "a set with no target gives an empty target block"


def test_same_key_gives_the_same_draw_and_fresh_arrays() -> None:
    first, y_first = sample_rows("wine", rng("draws:same"), 20, 4)
    second, y_second = sample_rows("wine", rng("draws:same"), 20, 4)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(y_first, y_second)
    assert first is not second
    first[:] = 0.0  # a caller may mutate its draw; the cached dataset must not change
    third, _ = sample_rows("wine", rng("draws:same"), 20, 4)
    np.testing.assert_array_equal(third, second)


def test_rows_and_columns_are_drawn_without_replacement() -> None:
    wine = load("wine")
    drawn, _ = sample_rows("wine", rng("draws:all"), wine.X.shape[0], wine.X.shape[1])
    got = sorted(tuple(sorted(column)) for column in drawn.T.tolist())
    assert got == sorted(tuple(sorted(column)) for column in wine.X.T.tolist())


def test_column_subset_has_the_requested_width() -> None:
    drawn, y = sample_rows("breast_cancer", rng("draws:cols"), 30, 3)
    assert drawn.shape == (30, 3)
    assert y.shape == (30,)
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}


def test_multi_output_target_stays_two_dimensional() -> None:
    _, y = sample_rows("linnerud", rng("draws:multi"), 7)
    assert y.shape == (7, 3)


def test_standardize_uses_the_whole_dataset() -> None:
    wine = load("wine")
    raw, _ = sample_rows("wine", rng("draws:z"), 12)
    scaled, _ = sample_rows("wine", rng("draws:z"), 12, standardize=True)
    expected = (raw - wine.X.mean(axis=0)) / wine.X.std(axis=0)
    np.testing.assert_allclose(scaled, expected, rtol=1e-12, atol=1e-12)


def test_standardize_maps_a_constant_column_to_zeros() -> None:
    digits = load("digits")
    constant = np.flatnonzero(digits.X.std(axis=0) == 0.0)
    assert constant.size == 3
    scaled, _ = sample_rows("digits", rng("draws:const"), 40, standardize=True)
    assert np.isfinite(scaled).all()
    np.testing.assert_array_equal(scaled[:, constant], 0.0)


@pytest.mark.parametrize(
    ("name", "n", "d"),
    [("wine", 0, None), ("wine", 179, None), ("wine", 5, 0), ("wine", 5, 14), ("wine", True, 2)],
)
def test_sizes_outside_the_dataset_are_refused(name: str, n: int, d: int | None) -> None:
    with pytest.raises(ValueError, match="must be"):
        sample_rows(name, rng("draws:bad"), n, d)


@pytest.mark.parametrize("name", ["co2", "penguins", "gsm8k", "gutenberg_walden", "nope"])
def test_non_numeric_ids_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match=r"numeric table|unknown dataset"):
        sample_rows(name, rng("draws:kind"), 3)


@pytest.mark.parametrize("name", ["co2", "elnino", "nile", "sunspots"])
def test_window_is_a_run_of_the_series(name: str) -> None:
    values = load_series(name)
    values = values[~np.isnan(values)]
    window = sample_window(name, rng(f"draws:window:{name}"), 24)
    assert window.shape == (24,)
    assert window.dtype == np.float64
    starts = [i for i in range(len(values) - 23) if np.array_equal(values[i : i + 24], window)]
    assert starts, "the window is not a run of consecutive values"


def test_window_can_keep_missing_values() -> None:
    series = load_series("co2")
    window = sample_window("co2", rng("draws:nan"), series.shape[0], keep_nan=True)
    np.testing.assert_array_equal(window, series)
    assert int(np.isnan(window).sum()) == 59


def test_window_is_deterministic_and_validated() -> None:
    first = sample_window("nile", rng("draws:w"), 30)
    second = sample_window("nile", rng("draws:w"), 30)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError, match="must be between"):
        sample_window("nile", rng("draws:w"), 101)
    with pytest.raises(ValueError, match="not a series"):
        sample_window("wine", rng("draws:w"), 5)
