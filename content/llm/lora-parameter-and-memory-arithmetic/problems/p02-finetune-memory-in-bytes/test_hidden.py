# Hidden tests (CONTENT_SPEC §6.4). Names describe the behaviour; messages name the kind of mistake
# and never print hidden inputs, shapes or expected values.
from __future__ import annotations

import functools
import random

import pytest
from submission import finetune_memory_bytes as _learner

from dogma_data import load

KEYS = ["base_weights", "adapter_weights", "adapter_gradients", "adam_states", "total"]


def _call(**kwargs):
    error = None
    try:
        out = _learner(**kwargs)
    except ValueError as exc:
        error = f"valid arguments were rejected with ValueError ({exc}): check the allowed sets"
    except Exception as exc:
        error = f"valid arguments raised {type(exc).__name__}: check integer handling"
    if error is not None:
        pytest.fail(error, pytrace=False)
    return out


def _reference(base_params, base_bits, lora_params, adapter_bits):
    """Each stored tensor rounded up to whole bytes, with an independent rounding formula."""

    def up(count, bits):
        return -((-count * bits) // 8)  # ceiling division on exact integers

    base = up(base_params, base_bits)
    weights = up(lora_params, adapter_bits)
    adam = 2 * 4 * lora_params
    return {
        "base_weights": base,
        "adapter_weights": weights,
        "adapter_gradients": weights,
        "adam_states": adam,
        "total": base + weights + weights + adam,
    }


@functools.cache
def _dataset_ends() -> tuple[tuple[int, int], ...]:
    """(features, outputs) of five real datasets: iris 4 and 3, wine 13 and 3, breast_cancer 30
    and 2, digits 64 and 10, diabetes 10 and 1 (a regression target)."""
    outputs = {"iris": 3, "wine": 3, "breast_cancer": 2, "digits": 10, "diabetes": 1}
    return tuple((int(load(name).X.shape[1]), classes) for name, classes in outputs.items())


def test_random_configurations_match_the_reference_term_by_term():
    rng = random.Random(77)  # noqa: S311 -- seeded synthetic configurations, not security
    for trial in range(300):
        args = {
            "base_params": rng.randint(0, 10**11),
            "base_bits": rng.choice([4, 8, 16, 32]),
            "lora_params": rng.randint(0, 10**9),
            "adapter_bits": rng.choice([16, 32]),
        }
        if trial % 2:
            # Every other model is a dense network for a real dataset: its features in, a
            # drawn hidden width, its outputs out. The adapter is a low-rank pair on the
            # first layer.
            features, outputs = rng.choice(_dataset_ends())
            hidden, rank = rng.randint(1, 4096), rng.randint(1, 16)
            args["base_params"] = features * hidden + hidden + hidden * outputs + outputs
            args["lora_params"] = rank * (features + hidden)
        got = _call(**args)
        want = _reference(**args)
        assert list(got) == KEYS, "the dictionary must have the five keys in the stated order"
        for key in KEYS:
            assert got[key] == want[key], (
                f"term {key} differs from the rounded-up reference: check the bit arithmetic"
            )


def test_four_bit_rounding_is_exact_for_every_odd_count():
    for count in range(1, 33):
        got = _call(base_params=count, base_bits=4, lora_params=0, adapter_bits=16)["base_weights"]
        assert got == (count + 1) // 2, "4-bit storage must round up to whole bytes at every count"


def test_adam_term_is_fp32_whatever_the_adapter_dtype_is():
    for adapter_bits in (16, 32):
        got = _call(base_params=0, base_bits=16, lora_params=1000, adapter_bits=adapter_bits)
        assert got["adam_states"] == 8000, (
            "Adam keeps two fp32 moments: 8 bytes per adapter number, at any adapter dtype"
        )


def test_gradients_take_the_same_bytes_as_the_adapter_weights():
    got = _call(base_params=5, base_bits=16, lora_params=777, adapter_bits=16)
    assert got["adapter_gradients"] == got["adapter_weights"], (
        "gradients must match the adapter dtype"
    )


def test_frozen_base_has_no_gradient_or_optimizer_term():
    got = _call(base_params=10**6, base_bits=8, lora_params=0, adapter_bits=16)
    assert got["adapter_gradients"] == 0 and got["adam_states"] == 0, (
        "the frozen base must not add gradient or optimizer bytes"
    )


def test_total_is_the_exact_sum_of_the_four_terms():
    got = _call(base_params=12_345_678_901, base_bits=4, lora_params=98_765_432, adapter_bits=32)
    parts = (
        got["base_weights"] + got["adapter_weights"] + got["adapter_gradients"] + got["adam_states"]
    )
    assert got["total"] == parts, "total must be the exact sum of the four stored terms"


def test_realistic_configuration_stays_an_exact_integer():
    got = _call(base_params=70 * 10**9, base_bits=4, lora_params=2 * 10**8, adapter_bits=16)
    assert all(isinstance(got[key], int) for key in KEYS), "every term must be a Python int"
    assert got["base_weights"] == 35 * 10**9, "70e9 numbers at 4 bits is exactly 35e9 bytes"
    assert got["adam_states"] == 16 * 10**8, "two fp32 moments for 2e8 numbers is 16e8 bytes"


def test_large_counts_do_not_lose_precision():
    base = 2**60 + 3
    got = _call(base_params=base, base_bits=16, lora_params=0, adapter_bits=16)
    assert got["base_weights"] == 2 * base, (
        "large parameter counts must stay exact; no float rounding"
    )


def test_zero_base_gives_no_base_bytes():
    got = _call(base_params=0, base_bits=4, lora_params=5, adapter_bits=16)
    assert got["base_weights"] == 0, "a zero base must contribute zero bytes"


@pytest.mark.parametrize(
    "overrides, why",
    [
        ({"base_bits": 2}, "an unsupported base bit width"),
        ({"base_bits": 12}, "a base bit width outside the set"),
        ({"adapter_bits": 8}, "an 8-bit adapter"),
        ({"adapter_bits": 4}, "a 4-bit adapter"),
        ({"base_params": -1}, "a negative base_params"),
        ({"lora_params": -5}, "a negative lora_params"),
        ({"base_params": True}, "a bool base_params"),
        ({"lora_params": False}, "a bool lora_params"),
        ({"base_params": 1.5}, "a float base_params"),
        ({"base_bits": True}, "a bool base_bits"),
        ({"adapter_bits": 16.0}, "a float adapter_bits"),
    ],
    ids=[
        "base-bits-2",
        "base-bits-12",
        "adapter-8",
        "adapter-4",
        "negative-base",
        "negative-lora",
        "bool-base",
        "bool-lora",
        "float-base",
        "bool-base-bits",
        "float-adapter-bits",
    ],
)
def test_invalid_arguments_raise_value_error(overrides, why):
    kwargs = {"base_params": 100, "base_bits": 16, "lora_params": 10, "adapter_bits": 16}
    kwargs.update(overrides)
    error = None
    try:
        _learner(**kwargs)
    except ValueError:
        return
    except Exception as exc:
        error = f"{why} must raise ValueError, received {type(exc).__name__}"
    pytest.fail(error or f"{why} was accepted instead of raising ValueError", pytrace=False)
