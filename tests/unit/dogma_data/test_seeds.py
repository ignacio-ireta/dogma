"""rng(key): deterministic, key-separated, salted by DOGMA_DATA_SALT."""

from __future__ import annotations

import numpy as np
import pytest

from dogma_data import rng
from dogma_data.seeds import SALT_ENV


def _draw(key: str) -> np.ndarray:
    return rng(key).standard_normal(8)


def test_same_key_gives_identical_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SALT_ENV, raising=False)
    np.testing.assert_array_equal(_draw("problem:purpose:0"), _draw("problem:purpose:0"))


def test_different_keys_give_different_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SALT_ENV, raising=False)
    assert not np.array_equal(_draw("problem:purpose:0"), _draw("problem:purpose:1"))
    assert not np.array_equal(_draw("problem:a:0"), _draw("problem:b:0"))


def test_salt_changes_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SALT_ENV, raising=False)
    unsalted = _draw("problem:purpose:0")
    monkeypatch.setenv(SALT_ENV, "audit-1")
    salted = _draw("problem:purpose:0")
    monkeypatch.setenv(SALT_ENV, "audit-2")
    other_salt = _draw("problem:purpose:0")
    assert not np.array_equal(unsalted, salted)
    assert not np.array_equal(salted, other_salt)


def test_salted_stream_is_still_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SALT_ENV, "audit-1")
    np.testing.assert_array_equal(_draw("problem:purpose:0"), _draw("problem:purpose:0"))


def test_empty_salt_equals_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SALT_ENV, raising=False)
    unset = _draw("problem:purpose:0")
    monkeypatch.setenv(SALT_ENV, "")
    empty = _draw("problem:purpose:0")
    np.testing.assert_array_equal(unset, empty)


@pytest.mark.parametrize("bad", ["", None, 3, b"key"])
def test_bad_key_raises_value_error(bad: object) -> None:
    with pytest.raises(ValueError):
        rng(bad)  # type: ignore[arg-type]
