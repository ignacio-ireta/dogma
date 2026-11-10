from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import initialize_dense as _learner


def initialize_dense(*args):
    try:
        result = _learner(*args)
    except Exception as exc:
        pytest.fail(f"valid input violated the computation contract: {type(exc).__name__}")
    assert isinstance(result, np.ndarray), "initialized weights must be a NumPy array"
    assert result.dtype == np.float64, "initialized weights must be float64"
    assert result.shape == (args[0], args[1]), "weights must use fan-in by fan-out layout"
    return result


def _close(got, want, message):
    assert isinstance(got, np.ndarray), "return type must be a NumPy array"
    assert got.dtype == np.float64, "output dtype must be float64"
    assert got.shape == np.shape(want), "output shape must match the contract"
    assert np.allclose(got, want, atol=1e-9, rtol=1e-9), message


def _invalid(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid input must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid input was accepted instead of raising ValueError")


def test_seeded_oracle_uses_raw_base_draws_and_independent_scaling():
    for rows, cols in ((1, 4), (5, 2), (4, 4)):
        for seed in (0, 19, 501):
            rng = np.random.default_rng(seed)
            unit = rng.random((rows, cols))
            a = np.sqrt(6 / (rows + cols))
            expected = (2 * unit - 1) * a
            _close(
                initialize_dense(rows, cols, "xavier_uniform", seed),
                expected,
                "Xavier distribution, fan scale or draw order is incorrect",
            )
            expected_normal = np.random.default_rng(seed).standard_normal((rows, cols)) * np.sqrt(
                2 / rows
            )
            _close(
                initialize_dense(rows, cols, "he_normal", seed),
                expected_normal,
                "He must use fan-in and normal base draws",
            )


def test_repeated_seed_is_independent_of_call_history_and_storage():
    first = initialize_dense(3, 4, "he_normal", 20)
    saved = first.copy()
    initialize_dense(9, 2, "xavier_uniform", 3)
    second = initialize_dense(3, 4, "he_normal", 20)
    _close(second, saved, "local seeded draws must not depend on call history")
    first[...] = 0
    _close(second, saved, "separate calls must not share output storage")


def test_global_numpy_rng_unchanged():
    before = np.random.get_state()
    initialize_dense(2, 3, "xavier_uniform", 7)
    initialize_dense(2, 3, "he_normal", 7)
    after = np.random.get_state()
    assert (
        before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    ), "initialization must leave global RNG state untouched"


def test_invalid_arguments():
    for bad in (0, -1, True, 2.0, np.int64(2)):
        _invalid(_learner, bad, 3, "he_normal", 0)
        _invalid(_learner, 3, bad, "he_normal", 0)
    for scheme in ("xavier", "he_uniform", "", None):
        _invalid(_learner, 2, 3, scheme, 0)
    for seed in (-1, True, 1.0, np.int64(1)):
        _invalid(_learner, 2, 3, "he_normal", seed)


def test_singleton_and_wide_output_scaling():
    _close(
        initialize_dense(1, 1, "xavier_uniform", 8),
        (2 * np.random.default_rng(8).random((1, 1)) - 1) * np.sqrt(3),
        "singleton fan scaling is incorrect",
    )
    normal = np.random.default_rng(8).standard_normal((1, 200))
    _close(
        initialize_dense(1, 200, "he_normal", 8),
        normal * np.sqrt(2),
        "He scale depends on input width, not output width",
    )


@settings(max_examples=30, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 9), st.integers(1, 9), st.integers(0, 5000))
def test_xavier_rescaling_recovers_same_uniform_base(rows, cols, seed):
    got = initialize_dense(rows, cols, "xavier_uniform", seed)
    a = np.sqrt(6 / (rows + cols))
    unit = (got / a + 1) / 2
    assert np.allclose(
        unit, np.random.default_rng(seed).random((rows, cols)), atol=1e-12, rtol=1e-12
    ), "Xavier transform must preserve the seeded unit-uniform draws"


def test_scheme_name_is_case_sensitive():
    # The Contract says the scheme is exactly one of two lowercase names.
    for scheme in ("XAVIER_UNIFORM", "He_Normal", "HE_NORMAL"):
        _invalid(_learner, 2, 3, scheme, 0)
