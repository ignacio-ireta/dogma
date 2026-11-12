# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import sinusoidal_encoding as _learner

from dogma_data import load_frame


def _table(max_len, d_model):
    """Call the learner; any exception on a valid input is reported as a named contract failure."""
    try:
        return _learner(max_len, d_model)
    except Exception as exc:
        pytest.fail(f"valid arguments violated the contract: raised {type(exc).__name__}")


def _reference(max_len, d_model):
    """Independent position-by-position evaluation of the Vaswani et al. (2017) formula."""
    out = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(d_model // 2):
            angle = pos / 10000 ** (2 * i / d_model)
            out[pos, 2 * i] = math.sin(angle)
            out[pos, 2 * i + 1] = math.cos(angle)
    return out


def _invalid(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument must raise ValueError, received {type(exc).__name__}")
    pytest.fail("invalid argument was accepted instead of raising ValueError")


def test_matches_position_by_position_reference_on_seeded_shapes():
    # The table is as long as a real sequence: the number of words of a GSM8K question.
    questions = load_frame("gsm8k")["question"].tolist()
    rng = np.random.default_rng(3)
    for _ in range(8):
        max_len = len(questions[int(rng.integers(0, len(questions)))].split())
        d_model = 2 * int(rng.integers(1, 20))
        got = _table(max_len, d_model)
        np.testing.assert_allclose(
            got.numpy(),
            _reference(max_len, d_model),
            atol=1e-9,
            rtol=1e-9,
            err_msg="table disagrees with the per-position formula: check angle and column order",
        )


def test_even_columns_are_sines_and_odd_columns_are_cosines_interleaved():
    got = _table(5, 8).numpy()
    ref = _reference(5, 8)
    np.testing.assert_allclose(
        got[:, 0::2],
        ref[:, 0::2],
        atol=1e-9,
        err_msg="even columns must hold sines: the table is interleaved, not split into blocks",
    )
    np.testing.assert_allclose(
        got[:, 1::2],
        ref[:, 1::2],
        atol=1e-9,
        err_msg="odd columns must hold cosines paired with the sine before them",
    )


def test_smallest_width_uses_unit_frequency_on_every_position():
    got = _table(7, 2).numpy()
    pos = np.arange(7, dtype=float)
    np.testing.assert_allclose(
        got[:, 0],
        np.sin(pos),
        atol=1e-9,
        err_msg="width-2 table: the sine must use the position itself as its angle",
    )
    np.testing.assert_allclose(
        got[:, 1],
        np.cos(pos),
        atol=1e-9,
        err_msg="width-2 table: the cosine must share the sine's angle",
    )


def test_last_frequency_at_wide_width_uses_exponent_close_to_one():
    d = 512
    got = _table(20, d).numpy()
    pos = np.arange(20, dtype=float)
    angle = pos / 10000 ** ((d - 2) / d)
    np.testing.assert_allclose(
        got[:, d - 2],
        np.sin(angle),
        atol=1e-9,
        err_msg="lowest frequency is wrong: the exponent must be 2i/d_model with the base 10000",
    )
    np.testing.assert_allclose(
        got[:, d - 1],
        np.cos(angle),
        atol=1e-9,
        err_msg="lowest frequency cosine is wrong: it must share the sine's angle",
    )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(min_value=1, max_value=60), st.integers(min_value=1, max_value=32))
def test_each_sine_cosine_pair_has_unit_norm(max_len, half):
    got = _table(max_len, 2 * half).numpy()
    norms = np.sqrt(got[:, 0::2] ** 2 + got[:, 1::2] ** 2)
    np.testing.assert_allclose(
        norms, 1.0, atol=1e-9, err_msg="each sine/cosine pair must lie on the unit circle"
    )


def test_first_row_is_zero_one_pattern():
    got = _table(3, 6).numpy()
    np.testing.assert_allclose(
        got[0],
        [0, 1, 0, 1, 0, 1],
        atol=1e-12,
        err_msg="position 0 must give sin(0)=0 in even columns and cos(0)=1 in odd columns",
    )


def test_pair_rotation_relates_positions_by_offset():
    # Each pair (sin, cos) at position p + k is a rotation of the pair at p by the angle
    # omega * k, where omega depends only on the pair's frequency. Check it for a fixed offset.
    d, k = 10, 3
    got = _table(12, d).numpy()
    for i in range(d // 2):
        omega = 10000 ** (-2 * i / d)
        c, s = math.cos(omega * k), math.sin(omega * k)
        for p in range(12 - k):
            sp, cp = got[p, 2 * i], got[p, 2 * i + 1]
            sq, cq = got[p + k, 2 * i], got[p + k, 2 * i + 1]
            np.testing.assert_allclose(
                [sq, cq],
                [sp * c + cp * s, cp * c - sp * s],
                atol=1e-9,
                err_msg="offset must rotate each sin/cos pair: check the per-pair frequency",
            )


def test_long_table_stays_finite_and_bounded():
    got = _table(3000, 64)
    assert torch.isfinite(got).all(), "long table must not contain NaN or infinity"
    assert float(got.abs().max()) <= 1.0 + 1e-12, "sines and cosines must stay within [-1, 1]"


def test_output_is_float64_with_requested_shape():
    got = _table(4, 6)
    assert got.dtype == torch.float64, "output dtype must be torch.float64"
    assert tuple(got.shape) == (4, 6), "output shape must be (max_len, d_model)"


def test_negative_length_raises_value_error():
    _invalid(_learner, -1, 4)


def test_non_integer_length_raises_value_error():
    _invalid(_learner, 2.5, 4)


@pytest.mark.parametrize("width", [3, 1], ids=["odd-three", "odd-one"])
def test_odd_width_raises_value_error(width):
    _invalid(_learner, 4, width)


@pytest.mark.parametrize("width", [0, -2], ids=["zero", "negative"])
def test_non_positive_width_raises_value_error(width):
    _invalid(_learner, 4, width)


def test_zero_length_table_has_zero_rows_and_requested_width():
    got = _table(0, 4)
    assert tuple(got.shape) == (0, 4), "max_len = 0 must return an empty table with d_model columns"


def test_zero_length_table_is_float64():
    got = _table(0, 4)
    assert got.dtype == torch.float64, "an empty table must still be torch.float64"


def test_each_call_returns_a_new_tensor():
    first = _table(3, 4)
    second = _table(3, 4)
    assert first is not second, "each call must create a new tensor, not return an existing one"
