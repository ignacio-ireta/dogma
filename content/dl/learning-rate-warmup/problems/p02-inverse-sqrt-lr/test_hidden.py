# Hidden tests: the oracle is a LambdaLR driven by a real optimizer. The scheduler passes its own
# counter, which starts at 0; the paper numbers updates from 1. A correct implementation must
# therefore agree with the scheduler only after the +1 shift that the test applies.
import itertools
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import inverse_sqrt_lr
from torch.optim.lr_scheduler import LambdaLR

REL = 1e-9


def _reference(step, d_model, warmup):
    """Vaswani eq. 3 written as a plain function of the 1-based update number."""
    return d_model**-0.5 * min(step**-0.5, step * warmup**-1.5)


def _close(got, want, message):
    if not math.isclose(got, want, rel_tol=REL, abs_tol=0.0):
        pytest.fail(message, pytrace=False)


def _rejects(call, message):
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(message, pytrace=False)
    pytest.fail(message, pytrace=False)


def test_full_run_matches_lambda_scheduler_with_one_based_counter():
    rng = torch.Generator().manual_seed(29)
    for _ in range(25):
        d_model = int(torch.randint(1, 1024, (1,), generator=rng))
        warmup = int(torch.randint(1, 60, (1,), generator=rng))
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda s, d=d_model, w=warmup: _reference(s + 1, d, w)
        )
        for step in range(1, 3 * warmup + 5):
            # Before the k-th update the scheduler has been stepped k-1 times, so its
            # counter equals step - 1. The rate for this update is therefore the paper's step.
            got = inverse_sqrt_lr(step, d_model, warmup)
            _close(
                got,
                optimizer.param_groups[0]["lr"],
                "rate differs from the scheduler: check whether step is numbered from 1 "
                "as in the paper",
            )
            optimizer.step()
            scheduler.step()


def test_peak_is_at_warmup_steps_with_value_d_to_minus_half_times_w_to_minus_half():
    for d_model, warmup in ((512, 4000), (64, 7), (1, 1)):
        peak = inverse_sqrt_lr(warmup, d_model, warmup)
        _close(
            peak,
            d_model**-0.5 * warmup**-0.5,
            "the peak must be d_model^-0.5 * warmup^-0.5: check the ramp exponent",
        )
        for step in (warmup - 1, warmup + 1):
            if step >= 1 and inverse_sqrt_lr(step, d_model, warmup) > peak * (1 + 1e-12):
                pytest.fail("a value next to the warmup peak exceeds it", pytrace=False)


def test_rate_rises_during_warmup_and_falls_afterwards():
    values = [inverse_sqrt_lr(s, 128, 9) for s in range(1, 60)]
    peak_index = values.index(max(values))
    if peak_index != 8:
        pytest.fail("peak must sit at step == warmup_steps (the ninth update)", pytrace=False)
    if any(b < a for a, b in itertools.pairwise(values[:9])):
        pytest.fail("rate must increase monotonically during warmup", pytrace=False)
    if any(b > a for a, b in itertools.pairwise(values[9:])):
        pytest.fail("rate must decrease monotonically after warmup", pytrace=False)


def test_decay_for_very_large_steps_uses_inverse_square_root():
    for step in (10**6, 10**12):
        _close(
            inverse_sqrt_lr(step, 256, 4000),
            256**-0.5 * step**-0.5,
            "after warmup the decay term must be step ** -0.5 for very large steps",
        )


def test_scaling_by_model_width_is_a_power_of_minus_half():
    for step in (1, 50, 500):
        _close(
            inverse_sqrt_lr(step, 16, 40),
            2 * inverse_sqrt_lr(step, 64, 40),
            "quadrupling d_model must halve the rate: check the d_model exponent",
        )


def test_single_step_warmup_peaks_immediately():
    _close(inverse_sqrt_lr(1, 100, 1), 0.1, "warmup of one update must peak at step 1")


@settings(max_examples=60, deadline=None, report_multiple_bugs=False)
@given(st.integers(1, 10**6), st.integers(1, 10**6), st.integers(1, 10**6))
def test_matches_the_paper_formula_for_random_sizes(step, d_model, warmup):
    _close(
        inverse_sqrt_lr(step, d_model, warmup),
        _reference(step, d_model, warmup),
        "rate differs from Vaswani et al. eq. 3: check the min() and the exponents",
    )


def test_extreme_sizes_keep_relative_accuracy():
    _close(
        inverse_sqrt_lr(10**9, 1, 10**6),
        _reference(10**9, 1, 10**6),
        "rate at extreme sizes differs from the formula: check the exponents and operation order",
    )
    _close(
        inverse_sqrt_lr(2, 10**6, 3),
        _reference(2, 10**6, 3),
        "rate at tiny values differs from the formula: check the exponents and operation order",
    )


def test_returns_a_python_float():
    if type(inverse_sqrt_lr(1, 1, 1)) is not float:
        pytest.fail("return a Python float, not an int or a tensor", pytrace=False)


def test_invalid_arguments_raise_value_error():
    _rejects(lambda: inverse_sqrt_lr(0, 4, 4), "step 0 must raise ValueError: steps start at 1")
    _rejects(lambda: inverse_sqrt_lr(-3, 4, 4), "negative step must raise ValueError")
    _rejects(lambda: inverse_sqrt_lr(True, 4, 4), "bool step must raise ValueError")
    _rejects(lambda: inverse_sqrt_lr(2.0, 4, 4), "float step must raise ValueError")
    _rejects(lambda: inverse_sqrt_lr(1, 0, 4), "d_model below 1 must raise ValueError")
    _rejects(lambda: inverse_sqrt_lr(1, 4, 0), "warmup_steps below 1 must raise ValueError")
    _rejects(lambda: inverse_sqrt_lr(1, 4.0, 4), "float d_model must raise ValueError")
