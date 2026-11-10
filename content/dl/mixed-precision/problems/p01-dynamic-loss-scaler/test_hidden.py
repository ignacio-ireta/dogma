# Hidden tests for DynamicLossScaler. The reference is torch.amp.GradScaler on CPU, driven with the
# same overflow sequence. Whether a step ran is read from the parameter (lr=1 moves it only when the
# step is applied). The remaining tests check the state machine exactly. Failure messages name the
# kind of mistake and never show hidden sequences or values.
from __future__ import annotations

import math
import random

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from submission import DynamicLossScaler


def _torch_run(flags, init_scale, growth, backoff, interval):
    """Run torch's CPU GradScaler on a sequence of found_inf flags. Return [(applied, scale)]."""
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=init_scale,
        growth_factor=growth,
        backoff_factor=backoff,
        growth_interval=interval,
    )
    results = []
    for flag in flags:
        # A fresh parameter and optimizer per step, so a moved parameter means the step ran.
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=1.0)
        scaler.scale(p.sum())  # torch needs the scale tensor created before step()
        p.grad = torch.tensor([float("inf") if flag else 1.0])
        scaler.step(opt)
        scaler.update()
        results.append((p.item() != 0.0, scaler.get_scale()))
    return results


def test_scale_and_steps_match_torch_grad_scaler():
    configs = [
        (1024.0, 2.0, 0.5, 3),
        (65536.0, 4.0, 0.25, 5),
        (1024.0, 2.0, 0.5, 1),
    ]
    rng = random.Random(7)  # noqa: S311 (seeded test data, not security)
    for init_scale, growth, backoff, interval in configs:
        flags = [rng.random() < 0.3 for _ in range(40)]
        reference = _torch_run(flags, init_scale, growth, backoff, interval)
        ours = DynamicLossScaler(
            init_scale=init_scale,
            growth_factor=growth,
            backoff_factor=backoff,
            growth_interval=interval,
        )
        for step, (flag, (applied_ref, scale_ref)) in enumerate(zip(flags, reference, strict=True)):
            applied = ours.update(found_inf=flag)
            assert applied == applied_ref, (
                "step decision differs from GradScaler: skip exactly when an overflow is found"
            )
            assert ours.scale == scale_ref, (
                f"scale differs from GradScaler at step {step}: check backoff and growth"
            )


def test_growth_counter_is_reset_by_an_overflow():
    scaler = DynamicLossScaler(init_scale=1.0, growth_interval=4)
    for _ in range(3):
        scaler.update(found_inf=False)
    scaler.update(found_inf=True)
    assert scaler.scale == 0.5, "an overflow must halve the scale"
    for _ in range(3):
        scaler.update(found_inf=False)
    assert scaler.scale == 0.5, (
        "growth counted good steps from before the overflow: the count must restart at 0"
    )
    scaler.update(found_inf=False)
    assert scaler.scale == 1.0, "growth must happen after a full interval following the overflow"


def test_growth_interval_one_doubles_every_good_step():
    scaler = DynamicLossScaler(init_scale=8.0, growth_interval=1)
    scaler.update(found_inf=False)
    assert scaler.scale == 16.0, "a good step must double the scale once the interval is reached"
    scaler.update(found_inf=True)
    assert scaler.scale == 8.0, "an overflow must halve the scale back down"
    assert scaler.update(found_inf=False) is True, "a good step must be reported as applied"
    assert scaler.scale == 16.0, "growth_interval 1 must double after each good step"


def test_default_growth_interval_is_two_thousand():
    scaler = DynamicLossScaler()
    assert scaler.scale == 2.0**16, "default init_scale must be 2**16"
    for _ in range(1999):
        scaler.update(found_inf=False)
    assert scaler.scale == 2.0**16, "the default interval is 2000: the scale grew too early"
    scaler.update(found_inf=False)
    assert scaler.scale == 2.0**17, "the scale must double at the 2000th good step"


def test_custom_growth_and_backoff_factors_are_applied():
    scaler = DynamicLossScaler(
        init_scale=1.0, growth_factor=3.0, backoff_factor=0.25, growth_interval=2
    )
    scaler.update(found_inf=False)
    scaler.update(found_inf=False)
    assert scaler.scale == 3.0, "growth must multiply by growth_factor, not by 2"
    scaler.update(found_inf=True)
    assert scaler.scale == 0.75, "backoff must multiply by backoff_factor"
    scaler.update(found_inf=False)
    scaler.update(found_inf=False)
    assert scaler.scale == 2.25, "growth after a backoff must multiply the scale by growth_factor"


def test_scale_stays_positive_after_a_long_run_of_overflows():
    scaler = DynamicLossScaler(init_scale=2.0**16)
    for _ in range(300):
        assert scaler.update(found_inf=True) is False, (
            "an overflow must be reported as a skipped step"
        )
    assert scaler.scale > 0.0, "the scale reached zero: backoff must never remove the scale"
    assert scaler.scale == 2.0**16 * 0.5**300, "backoff must halve on every overflow"
    assert scaler.good_steps == 0, "overflows must keep the good-step count at zero"


def test_state_is_per_instance_not_shared_between_scalers():
    a = DynamicLossScaler(init_scale=4.0, growth_interval=2)
    b = DynamicLossScaler(init_scale=4.0, growth_interval=2)
    a.update(found_inf=False)
    a.update(found_inf=False)
    assert a.scale == 8.0, "good steps on one scaler must grow only that scaler's scale"
    assert b.scale == 4.0 and b.good_steps == 0, "a second scaler saw the first one's state"


def test_types_are_float_scale_int_count_and_bool_decision():
    scaler = DynamicLossScaler(init_scale=2**10, growth_interval=2)
    decision = scaler.update(found_inf=False)
    assert isinstance(scaler.scale, float), "scale must be a Python float"
    assert isinstance(scaler.good_steps, int), "good_steps must be an int"
    assert isinstance(decision, bool), "update must return a bool"


def _problem_or_none(make) -> str | None:
    try:
        make()
    except ValueError:
        return None
    except Exception as exc:  # reported after the try block
        return f"raised {type(exc).__name__} instead of ValueError"
    return "accepted an invalid argument instead of raising ValueError"


def test_invalid_arguments_raise_value_error():
    checks = [
        lambda: DynamicLossScaler(init_scale=float("inf")),
        lambda: DynamicLossScaler(init_scale=float("nan")),
        lambda: DynamicLossScaler(growth_factor=1.0),
        lambda: DynamicLossScaler(growth_factor=0.5),
        lambda: DynamicLossScaler(backoff_factor=0.0),
        lambda: DynamicLossScaler(backoff_factor=1.0),
        lambda: DynamicLossScaler(backoff_factor=-0.5),
        lambda: DynamicLossScaler(growth_interval=0),
        lambda: DynamicLossScaler(growth_interval=2.5),
        lambda: DynamicLossScaler(init_scale=True),
        lambda: DynamicLossScaler(growth_interval=True),
    ]
    for make in checks:
        problem = _problem_or_none(make)
        assert problem is None, f"constructor argument not checked: {problem}"


@settings(max_examples=120, deadline=None, derandomize=True, report_multiple_bugs=False)
@given(
    flags=st.lists(st.booleans(), min_size=1, max_size=60),
    interval=st.integers(min_value=1, max_value=6),
)
def test_property_every_transition_follows_the_update_rule(flags: list[bool], interval: int):
    scaler = DynamicLossScaler(
        init_scale=2.0**12, growth_factor=2.0, backoff_factor=0.5, growth_interval=interval
    )
    for flag in flags:
        prev_scale, prev_good = scaler.scale, scaler.good_steps
        applied = scaler.update(found_inf=flag)
        assert applied == (not flag), "the step decision must be the negation of found_inf"
        if flag:
            assert scaler.scale == prev_scale * 0.5 and scaler.good_steps == 0, (
                "overflow transition wrong: halve the scale and reset the count"
            )
        elif prev_good + 1 == interval:
            assert scaler.scale == prev_scale * 2.0 and scaler.good_steps == 0, (
                "growth transition wrong: double the scale when the count reaches the interval"
            )
        else:
            assert scaler.scale == prev_scale and scaler.good_steps == prev_good + 1, (
                "good-step transition wrong: count the step and leave the scale unchanged"
            )
        assert 0 <= scaler.good_steps < interval, "good_steps must stay below growth_interval"
        assert math.isfinite(scaler.scale) and scaler.scale > 0.0, (
            "the loss scale must stay finite and positive"
        )


def test_scale_underflows_to_zero_after_more_than_1090_overflows():
    scaler = DynamicLossScaler(init_scale=2.0**16)
    for _ in range(1100):
        scaler.update(found_inf=True)
    # 2**16 * 0.5**1100 = 2**-1084, which is below half the smallest positive float (2**-1074),
    # so every halving is exact until the last one rounds to 0.0.
    assert scaler.scale == 0.0, (
        "the class must set no floor: after more than 1090 overflows the scale is 0.0"
    )
