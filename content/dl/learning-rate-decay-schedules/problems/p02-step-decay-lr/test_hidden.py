# Hidden tests: the oracle is torch's StepLR, driven through a real optimizer.
import math
import signal

import pytest
import torch
from submission import step_decay_lr
from torch.optim.lr_scheduler import StepLR

REL = 1e-9
ABS = 1e-12
SLOW_CALL_SECONDS = 2.0


class _TooSlow(BaseException):
    """Raised by the alarm. A BaseException, so a learner's broad `except` cannot hide it."""


def _returns_within(seconds, call):
    """Return call()'s value, or None when it does not return within `seconds` (Unix only)."""
    if not hasattr(signal, "SIGALRM"):
        return call()

    def _alarm(signum, frame):
        raise _TooSlow

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return call()
    except _TooSlow:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _scheduler_run(base_lr, step_size, gamma, updates):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=base_lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
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


def test_full_run_matches_step_lr_scheduler_on_seeded_settings():
    rng = torch.Generator().manual_seed(7)
    for _ in range(30):
        step_size = int(torch.randint(1, 9, (1,), generator=rng))
        gamma = float(torch.rand((), generator=rng) * 0.9 + 0.05)
        base_lr = float(torch.rand((), generator=rng) + 1e-3)
        updates = int(torch.randint(1, 40, (1,), generator=rng))
        expected = _scheduler_run(base_lr, step_size, gamma, updates)
        for step, want in enumerate(expected):
            _close(
                step_decay_lr(step, base_lr, step_size, gamma),
                want,
                "rate differs from the step-decay scheduler: check the block count "
                "or the drop boundary",
            )


def test_drop_happens_exactly_at_the_block_boundary_on_both_sides():
    for step_size in (1, 4, 10):
        _close(
            step_decay_lr(step_size - 1, 1.0, step_size, 0.1),
            1.0,
            "the rate must not drop before step_size updates have passed",
        )
        _close(
            step_decay_lr(step_size, 1.0, step_size, 0.1),
            0.1,
            "the rate must drop once step_size updates have passed",
        )


def test_gamma_of_one_is_constant_for_every_step():
    for step in (0, 1, 50, 999):
        _close(step_decay_lr(step, 0.7, 3, 1.0), 0.7, "gamma of one must leave the rate unchanged")


def test_huge_step_count_is_evaluated_directly():
    # The contract requires constant time. Iterating over a trillion steps cannot finish in
    # the runner's budget, so a time limit turns that mistake into a message, not a timeout.
    # Underflow to a tiny value is acceptable here, so compare with an absolute tolerance only.
    step = 10**12
    got = _returns_within(SLOW_CALL_SECONDS, lambda: step_decay_lr(step, 0.5, 3, 0.9))
    if got is None:
        pytest.fail(
            "a very large step count did not return in time: compute floor(step / step_size) "
            "and one power of gamma; do not loop over the steps (the contract requires "
            "constant time)",
            pytrace=False,
        )
    want = 0.5 * 0.9 ** (step // 3)
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-300):
        pytest.fail("large step counts must use the block count, not iteration", pytrace=False)


def test_small_base_rate_and_small_gamma_stay_accurate():
    got = step_decay_lr(3, 1e-10, 1, 1e-3)
    want = 1e-10 * 1e-3**3
    if not math.isclose(got, want, rel_tol=1e-9, abs_tol=0.0):
        pytest.fail("tiny base rates or factors lose relative accuracy", pytrace=False)


def test_returns_a_python_float():
    if type(step_decay_lr(0, 1, 1, 1)) is not float:
        pytest.fail("return a Python float, not an int or a tensor", pytrace=False)


def test_invalid_arguments_raise_value_error():
    _rejects(lambda: step_decay_lr(-1, 0.1, 2, 0.5), "negative step must raise ValueError")
    _rejects(lambda: step_decay_lr(False, 0.1, 2, 0.5), "bool step must raise ValueError")
    _rejects(lambda: step_decay_lr(2, 0.1, 0, 0.5), "step_size below 1 must raise ValueError")
    _rejects(lambda: step_decay_lr(2, 0.1, 2.0, 0.5), "float step_size must raise ValueError")
    _rejects(lambda: step_decay_lr(2, 0.1, True, 0.5), "bool step_size must raise ValueError")
    _rejects(lambda: step_decay_lr(2, -0.1, 2, 0.5), "negative base_lr must raise ValueError")
    _rejects(lambda: step_decay_lr(2, 0.1, 2, 0.0), "gamma of zero must raise ValueError")
    _rejects(lambda: step_decay_lr(2, 0.1, 2, 1.5), "gamma above one must raise ValueError")
    _rejects(lambda: step_decay_lr(2, 0.1, 2, float("inf")), "infinite gamma must raise ValueError")


def test_zero_base_rate_raises_value_error():
    _rejects(
        lambda: step_decay_lr(0, 0.0, 2, 0.5),
        "a base_lr of zero is not positive and must raise ValueError",
    )
