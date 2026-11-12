# Hidden tests: the oracle is torch's CosineAnnealingLR, driven through a real optimizer.
# Names describe the behaviour; assertion messages name the kind of mistake.
import itertools
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import cosine_annealing_lr
from torch.optim.lr_scheduler import CosineAnnealingLR

from dogma_data import load

REL = 1e-9
ABS = 1e-12


def _scheduler_run(base_lr, t_max, eta_min):
    """Rates after 0..t_max scheduler updates, with the optimizer stepped before each update."""
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=base_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(t_max):
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


def test_full_run_matches_cosine_annealing_scheduler_on_seeded_settings():
    # The schedule lasts one epoch over a real dataset; the two rates are drawn.
    steps = _steps_per_epoch()
    rng = torch.Generator().manual_seed(2026)
    for _ in range(40):
        t_max = steps[int(torch.randint(0, len(steps), (1,), generator=rng))]
        base_lr = float(torch.rand((), generator=rng) * 2 + 1e-3)
        eta_min = float(torch.rand((), generator=rng) * base_lr)
        expected = _scheduler_run(base_lr, t_max, eta_min)
        for step, want in enumerate(expected):
            got = cosine_annealing_lr(step, base_lr, t_max, eta_min)
            _close(
                got,
                want,
                "rate differs from the cosine-annealing scheduler: check the half-cosine "
                "phase, the t_max scaling or the eta_min blend",
            )


def test_rate_reaches_floor_exactly_at_t_max_not_one_step_later():
    for t_max in (1, 2, 5, 17):
        _close(
            cosine_annealing_lr(t_max, 0.3, t_max, 0.02),
            0.02,
            "the floor must be reached at step t_max: check the period length",
        )


def test_steps_after_t_max_hold_the_floor_instead_of_rising():
    for step in (5, 6, 20, 1000):
        _close(
            cosine_annealing_lr(step, 0.1, 4, 0.001),
            0.001,
            "steps past t_max must hold eta_min; the cosine must not restart",
        )


def test_single_update_schedule_jumps_from_base_to_floor():
    _close(cosine_annealing_lr(0, 0.5, 1, 0.05), 0.5, "step 0 must return base_lr")
    _close(cosine_annealing_lr(1, 0.5, 1, 0.05), 0.05, "one-update schedule must end at eta_min")


def test_equal_floor_and_base_rate_gives_constant_schedule():
    for step in range(0, 12):
        _close(
            cosine_annealing_lr(step, 0.25, 9, 0.25),
            0.25,
            "eta_min equal to base_lr must give a constant rate",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(1, 80),
    st.floats(1e-6, 10.0, allow_nan=False, allow_infinity=False),
    st.floats(0.0, 1.0, allow_nan=False),
)
def test_rate_is_symmetric_about_the_midpoint(t_max, base_lr, fraction):
    eta_min = base_lr * fraction
    for step in range(t_max + 1):
        left = cosine_annealing_lr(step, base_lr, t_max, eta_min)
        right = cosine_annealing_lr(t_max - step, base_lr, t_max, eta_min)
        _close(
            left + right,
            base_lr + eta_min,
            "rates at symmetric steps must sum to base_lr + eta_min: check the cosine sign",
        )


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 80), st.floats(1e-6, 10.0, allow_nan=False, allow_infinity=False))
def test_rate_never_increases_within_one_period(t_max, base_lr):
    values = [cosine_annealing_lr(step, base_lr, t_max) for step in range(t_max + 1)]
    if any(later > earlier + 1e-15 for earlier, later in itertools.pairwise(values)):
        pytest.fail(
            "rate rose within the period: the phase or the sign of the cosine is wrong",
            pytrace=False,
        )


def test_extreme_magnitudes_keep_relative_accuracy():
    for base_lr, eta_min in ((1e-12, 0.0), (1e6, 1e4)):
        expected = _scheduler_run(base_lr, 7, eta_min)
        for step, want in enumerate(expected):
            got = cosine_annealing_lr(step, base_lr, 7, eta_min)
            if not math.isclose(got, want, rel_tol=1e-9, abs_tol=0.0):
                pytest.fail(
                    "rate at an extreme base rate differs from the scheduler: check the "
                    "formula, or a blend that cancels when the rate is tiny",
                    pytrace=False,
                )


def test_returns_a_python_float_for_integer_inputs():
    value = cosine_annealing_lr(0, 1, 2, 0)
    if type(value) is not float:
        pytest.fail("return a Python float, not an int or a tensor", pytrace=False)


def test_invalid_arguments_raise_value_error():
    _rejects(lambda: cosine_annealing_lr(-1, 0.1, 4), "negative step must raise ValueError")
    _rejects(lambda: cosine_annealing_lr(True, 0.1, 4), "bool step must raise ValueError")
    _rejects(lambda: cosine_annealing_lr(1.0, 0.1, 4), "float step must raise ValueError")
    _rejects(lambda: cosine_annealing_lr(0, 0.1, 0), "t_max below 1 must raise ValueError")
    _rejects(lambda: cosine_annealing_lr(0, 0.0, 4), "base_lr of zero must raise ValueError")
    _rejects(lambda: cosine_annealing_lr(0, float("nan"), 4), "NaN base_lr must raise ValueError")
    _rejects(
        lambda: cosine_annealing_lr(0, 0.1, 4, eta_min=0.2),
        "eta_min above base_lr must raise ValueError",
    )
    _rejects(
        lambda: cosine_annealing_lr(0, 0.1, 4, eta_min=-0.01),
        "negative eta_min must raise ValueError",
    )


def test_non_integer_t_max_raises_value_error():
    _rejects(
        lambda: cosine_annealing_lr(0, 0.1, 4.0),
        "a float t_max must raise ValueError: t_max must be an int",
    )
    _rejects(
        lambda: cosine_annealing_lr(1, 0.1, 2.5),
        "a fractional t_max must raise ValueError: t_max must be an int",
    )


def test_infinite_base_lr_raises_value_error():
    _rejects(
        lambda: cosine_annealing_lr(0, float("inf"), 4),
        "infinite base_lr must raise ValueError: base_lr must be finite",
    )
