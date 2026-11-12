# Hidden tests: the oracle is a LambdaLR driven by a real optimizer. The lambda returns the
# multiplier of the reference piecewise schedule, so the test checks the step counting the
# scheduler uses as well as the formula. Names describe behaviour; messages name the mistake.
import itertools
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import warmup_cosine_lr
from torch.optim.lr_scheduler import LambdaLR

from dogma_data import load

REL = 1e-9
ABS = 1e-12


def _reference_multiplier(step, warmup, total, min_ratio):
    """Reference factor relative to base_lr: ramp, then cosine, written independently."""
    if step < warmup:
        return step / warmup
    if step >= total:
        return min_ratio
    progress = (step - warmup) / (total - warmup)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _scheduler_run(base_lr, warmup, total, min_lr, updates):
    """Rates after 0..updates scheduler steps, with the optimizer stepped before each one."""
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=base_lr)
    min_ratio = min_lr / base_lr
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda s: _reference_multiplier(s, warmup, total, min_ratio)
    )
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(updates):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    return rates


def _close(got, want, message):
    if not math.isclose(got, want, rel_tol=REL, abs_tol=ABS):
        pytest.fail(message, pytrace=False)


def _rejects(call, message):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(message, pytrace=False)
    pytest.fail(message, pytrace=False)


def _steps_per_epoch():
    """Updates in one epoch over a real dataset: its rows over a batch size, rounded up.

    Five datasets (iris 150 rows, wine 178, diabetes 442, breast_cancer 569, digits 1,797) at
    batch sizes 16, 32 and 64 give fifteen run lengths from 3 to 113 updates.
    """
    steps = []
    for name in ("iris", "wine", "diabetes", "breast_cancer", "digits"):
        rows = load(name).X.shape[0]
        steps.extend(math.ceil(rows / batch) for batch in (16, 32, 64))
    return steps


def test_full_run_matches_warmup_then_cosine_scheduler_on_seeded_settings():
    # After the warmup the schedule lasts one epoch over a real dataset. The warmup length
    # and the two rates are drawn.
    steps = _steps_per_epoch()
    rng = torch.Generator().manual_seed(11)
    for _ in range(30):
        warmup = int(torch.randint(0, 12, (1,), generator=rng))
        total = warmup + steps[int(torch.randint(0, len(steps), (1,), generator=rng))]
        base_lr = float(torch.rand((), generator=rng) * 2 + 1e-3)
        min_lr = float(torch.rand((), generator=rng) * base_lr)
        expected = _scheduler_run(base_lr, warmup, total, min_lr, total + 3)
        for step, want in enumerate(expected):
            _close(
                warmup_cosine_lr(step, base_lr, warmup, total, min_lr),
                want,
                "rate differs from the warmup-then-cosine schedule: check the ramp, "
                "the cosine start at warmup_steps, or the floor at total_steps",
            )


def test_ramp_starts_at_zero_and_rises_linearly():
    base, warmup = 0.8, 5
    for step in range(warmup):
        _close(
            warmup_cosine_lr(step, base, warmup, 40, 0.0),
            base * step / warmup,
            "ramp must rise linearly from zero: check the warmup denominator",
        )


def test_boundary_at_warmup_steps_is_exactly_base_rate_without_a_dip():
    for warmup in (1, 3, 10):
        _close(
            warmup_cosine_lr(warmup, 0.5, warmup, warmup + 20, 0.01),
            0.5,
            "at step == warmup_steps the rate must be base_lr, with no dip or second peak",
        )


def test_rate_reaches_floor_at_total_steps_and_stays_there():
    for step in (20, 21, 500, 10**9):
        _close(
            warmup_cosine_lr(step, 0.5, 4, 20, 0.01),
            0.01,
            "steps at or after total_steps must return min_lr, not a rising cosine",
        )


def test_zero_warmup_starts_at_the_base_rate():
    _close(
        warmup_cosine_lr(0, 0.3, 0, 10, 0.03), 0.3, "with no warmup, step 0 must already be base_lr"
    )


def test_one_step_warmup_has_zero_then_base_rate():
    _close(
        warmup_cosine_lr(0, 0.9, 1, 6, 0.0), 0.0, "a one-step ramp still starts from zero at step 0"
    )
    _close(
        warmup_cosine_lr(1, 0.9, 1, 6, 0.0),
        0.9,
        "after a one-step ramp the next rate must be base_lr",
    )


def test_equal_floor_and_base_rate_gives_constant_rate_after_warmup():
    for step in range(3, 25):
        _close(
            warmup_cosine_lr(step, 0.2, 3, 20, 0.2),
            0.2,
            "min_lr equal to base_lr must give a constant rate after warmup",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(1, 20),
    st.integers(1, 80),
    st.floats(1e-6, 10.0, allow_nan=False, allow_infinity=False),
)
def test_rate_rises_then_falls_around_the_peak(warmup, length, base_lr):
    total = warmup + length
    values = [warmup_cosine_lr(s, base_lr, warmup, total, 0.0) for s in range(total + 2)]
    rising = values[: warmup + 1]
    falling = values[warmup:]
    if any(b < a - 1e-15 for a, b in itertools.pairwise(rising)):
        pytest.fail("rate fell during warmup: the ramp must be nondecreasing", pytrace=False)
    if any(b > a + 1e-15 for a, b in itertools.pairwise(falling)):
        pytest.fail("rate rose after warmup: the cosine must be nonincreasing", pytrace=False)


def test_extreme_magnitudes_keep_relative_accuracy():
    for base_lr, min_lr in ((1e-12, 1e-15), (1e6, 1e3)):
        expected = _scheduler_run(base_lr, 3, 9, min_lr, 10)
        for step, want in enumerate(expected):
            got = warmup_cosine_lr(step, base_lr, 3, 9, min_lr)
            if not math.isclose(got, want, rel_tol=1e-9, abs_tol=0.0):
                pytest.fail(
                    "rate at an extreme base rate differs from the scheduler: check the "
                    "formula, or a blend that cancels when the rate is tiny",
                    pytrace=False,
                )


def test_returns_a_python_float():
    if type(warmup_cosine_lr(0, 1, 0, 2, 0)) is not float:
        pytest.fail("return a Python float, not an int or a tensor", pytrace=False)


def test_invalid_arguments_raise_value_error():
    _rejects(lambda: warmup_cosine_lr(-1, 1.0, 2, 10, 0.1), "negative step must raise ValueError")
    _rejects(lambda: warmup_cosine_lr(True, 1.0, 2, 10, 0.1), "bool step must raise ValueError")
    _rejects(lambda: warmup_cosine_lr(1.5, 1.0, 2, 10, 0.1), "float step must raise ValueError")
    _rejects(
        lambda: warmup_cosine_lr(0, 1.0, -1, 10, 0.1), "negative warmup_steps must raise ValueError"
    )
    _rejects(
        lambda: warmup_cosine_lr(0, 1.0, 4, 4, 0.1),
        "total_steps equal to warmup_steps must raise ValueError",
    )
    _rejects(
        lambda: warmup_cosine_lr(0, 1.0, 4, 3, 0.1),
        "total_steps below warmup_steps must raise ValueError",
    )
    _rejects(lambda: warmup_cosine_lr(0, 0.0, 2, 10, 0.0), "zero base_lr must raise ValueError")
    _rejects(
        lambda: warmup_cosine_lr(0, 1.0, 2, 10, 1.5), "min_lr above base_lr must raise ValueError"
    )
    _rejects(lambda: warmup_cosine_lr(0, 1.0, 2, 10, -0.1), "negative min_lr must raise ValueError")


def test_infinite_base_lr_raises_value_error():
    # Contract: base_lr must be finite and positive, so an infinite peak rate is invalid.
    _rejects(
        lambda: warmup_cosine_lr(1, math.inf, 2, 10, 0.0),
        "infinite base_lr must raise ValueError",
    )
