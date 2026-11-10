# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import math
import random

import pytest
from submission import lora_budget as _learner

ALL = ("q", "k", "v", "o", "gate", "up", "down")
ATTN = ("q", "k", "v", "o")
MLP = ("gate", "up", "down")


def _call(**kwargs):
    error = None
    try:
        out = _learner(**kwargs)
    except ValueError as exc:
        error = f"valid configuration was rejected with ValueError ({exc}): check the rules"
    except Exception as exc:
        error = f"valid configuration raised {type(exc).__name__}: check argument handling"
    if error is not None:
        pytest.fail(error, pytrace=False)
    return out


def _reference(n_layers, d_model, d_ff, rank, targets):
    """Plain-Python sum over matrices: rank * (d_in + d_out), times the number of layers."""
    total = 0
    for name in targets:
        if name in ATTN:
            d_in, d_out = d_model, d_model
        elif name in ("gate", "up"):
            d_in, d_out = d_model, d_ff
        else:
            d_in, d_out = d_ff, d_model
        total += rank * (d_in + d_out)
    return n_layers * total


def _expect_value_error(kwargs, message):
    error = None
    try:
        _learner(**kwargs)
    except ValueError:
        return
    except Exception as exc:
        error = f"invalid configuration must raise ValueError, received {type(exc).__name__}"
    pytest.fail(error or message, pytrace=False)


def test_random_configurations_match_the_per_matrix_reference():
    rng = random.Random(2024)  # noqa: S311 -- seeded synthetic configurations, not security
    for _ in range(200):
        n_layers = rng.randint(1, 40)
        d_model = rng.randint(1, 64)
        d_ff = rng.randint(1, 128)
        rank = rng.randint(1, 16)
        targets = tuple(rng.sample(ALL, rng.randint(1, len(ALL))))
        trainable, fraction = _call(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            rank=rank,
            targets=targets,
            base_params=10**9,
        )
        want = _reference(n_layers, d_model, d_ff, rank, targets)
        assert trainable == want, (
            "trainable count differs from the per-matrix sum: check d_in + d_out and layers"
        )
        assert math.isclose(fraction, want / 10**9, rel_tol=1e-12), (
            "fraction must be trainable / base_params"
        )


def test_adding_a_target_adds_exactly_its_own_parameters():
    base, _ = _call(n_layers=4, d_model=10, d_ff=20, rank=3, targets=("q",), base_params=1)
    more, _ = _call(n_layers=4, d_model=10, d_ff=20, rank=3, targets=("q", "down"), base_params=1)
    assert more - base == 4 * 3 * (20 + 10), (
        "the down matrix must add rank * (d_ff + d_model) per layer"
    )


def test_rank_one_gives_one_unit_per_input_plus_output_width():
    trainable, _ = _call(n_layers=1, d_model=5, d_ff=None, rank=1, targets=("v",), base_params=50)
    assert trainable == 10, "rank one on a 5 by 5 matrix holds 5 + 5 = 10 numbers"


def test_rank_above_the_width_is_counted_by_the_formula_not_capped():
    trainable, _ = _call(n_layers=1, d_model=4, d_ff=None, rank=9, targets=("k",), base_params=10)
    assert trainable == 72, "the rank is not capped at the width: 9 * (4 + 4) = 72"


def test_mlp_orientation_does_not_change_the_count_when_d_ff_is_below_d_model():
    trainable, _ = _call(
        n_layers=1, d_model=12, d_ff=3, rank=2, targets=("gate", "down"), base_params=1
    )
    assert trainable == 2 * (12 + 3) + 2 * (3 + 12), "gate and down must use d_ff on the right side"


def test_realistic_size_stays_exact_beyond_float32_precision():
    n_layers, d_model, d_ff, rank = 80, 8192, 28672, 16
    trainable, _ = _call(
        n_layers=n_layers,
        d_model=d_model,
        d_ff=d_ff,
        rank=rank,
        targets=ATTN + MLP,
        base_params=70 * 10**9,
    )
    want = n_layers * rank * (4 * (2 * d_model) + 3 * (d_model + d_ff))
    assert trainable == want, "a realistic-size count must stay an exact integer"
    assert isinstance(trainable, int), "trainable must be a Python int, not a float"


def test_target_order_does_not_change_the_result():
    a, _ = _call(n_layers=2, d_model=6, d_ff=9, rank=2, targets=("up", "q", "down"), base_params=1)
    b, _ = _call(n_layers=2, d_model=6, d_ff=9, rank=2, targets=("down", "up", "q"), base_params=1)
    assert a == b, "the count must not depend on the order of targets"


def test_fraction_is_trainable_over_base_params_as_a_float():
    trainable, fraction = _call(
        n_layers=2, d_model=8, d_ff=None, rank=2, targets=("q", "v"), base_params=3000
    )
    assert isinstance(fraction, float), "fraction must be a float"
    assert math.isclose(fraction, trainable / 3000, rel_tol=1e-12), (
        "fraction must be trainable / base_params"
    )


def test_targets_list_is_accepted_as_well_as_a_tuple():
    a, _ = _call(n_layers=1, d_model=4, d_ff=None, rank=1, targets=["q", "k"], base_params=1)
    b, _ = _call(n_layers=1, d_model=4, d_ff=None, rank=1, targets=("q", "k"), base_params=1)
    assert a == b, "a list of targets must give the same count as a tuple"


def test_targets_are_not_modified():
    targets = ["q", "gate"]
    _call(n_layers=1, d_model=4, d_ff=8, rank=1, targets=targets, base_params=1)
    assert targets == ["q", "gate"], "the targets list was modified in place"


@pytest.mark.parametrize(
    "overrides, why",
    [
        ({"targets": ("q", "q")}, "a repeated target"),
        ({"targets": ("q", "wq")}, "an unknown target name"),
        ({"targets": ()}, "an empty target set"),
        ({"targets": "q"}, "targets given as a bare string"),
        ({"targets": ("up",), "d_ff": None}, "an MLP target with d_ff missing"),
        ({"n_layers": True}, "a bool n_layers"),
        ({"d_model": 2.0}, "a float d_model"),
        ({"rank": 0}, "a rank of zero"),
        ({"rank": -1}, "a negative rank"),
        ({"base_params": 0}, "a zero base_params"),
        ({"d_model": False}, "a bool d_model"),
        ({"n_layers": 0}, "zero layers"),
    ],
    ids=[
        "repeated-target",
        "unknown-target",
        "empty-targets",
        "string-targets",
        "missing-d-ff",
        "bool-layers",
        "float-width",
        "zero-rank",
        "negative-rank",
        "zero-base",
        "bool-width",
        "zero-layers",
    ],
)
def test_invalid_configuration_raises_value_error(overrides, why):
    kwargs = {
        "n_layers": 2,
        "d_model": 8,
        "d_ff": 16,
        "rank": 2,
        "targets": ("q", "v"),
        "base_params": 100,
    }
    kwargs.update(overrides)
    _expect_value_error(kwargs, f"{why} was accepted instead of raising ValueError")


def test_d_ff_may_be_none_when_no_mlp_target_is_counted():
    trainable, _ = _call(n_layers=1, d_model=4, d_ff=None, rank=1, targets=("o",), base_params=1)
    assert trainable == 8, "attention-only budgets must not need d_ff"


def test_unused_d_ff_is_not_checked_when_no_mlp_target_is_counted():
    trainable, _ = _call(n_layers=1, d_model=4, d_ff=0, rank=1, targets=("q",), base_params=1)
    assert trainable == 8, "d_ff must be ignored, not validated, when no MLP target is counted"


def test_true_as_d_model_is_refused_not_counted_as_one():
    # The Contract says d_model is an int and not a bool. True equals 1, so accepting it
    # would return a budget for a one-wide model. The call must raise instead.
    _expect_value_error(
        {
            "n_layers": 1,
            "d_model": True,
            "d_ff": None,
            "rank": 1,
            "targets": ("q",),
            "base_params": 1,
        },
        "a bool d_model was accepted instead of raising ValueError",
    )
