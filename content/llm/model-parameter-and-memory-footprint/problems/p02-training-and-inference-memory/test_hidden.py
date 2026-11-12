# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show hidden
# arguments or expected totals. The oracle is a hand-worked table of bytes per parameter for every
# combination the contract allows. It is written out as numbers, not as the solution's formula.
from __future__ import annotations

import pytest
from submission import estimate_memory

from dogma_data import load

TERMS = ("weights", "gradients", "master_weights", "optimizer_states", "activations", "total")

# (mode, dtype, optimizer, master) -> bytes per parameter, worked by hand:
#   weights (b) + gradients (b, training) + master (4, if lower precision and on)
#   + optimizer states (4 each, sgd 0, momentum 1, adam and adamw 2).
BYTES_PER_PARAM = {
    ("inference", "fp32", "none", False): 4,
    ("inference", "fp16", "none", False): 2,
    ("inference", "bf16", "none", False): 2,
    ("inference", "fp8", "none", False): 1,
    ("inference", "int8", "none", False): 1,
    ("training", "fp32", "sgd", False): 8,  # 4 + 4 + 0
    ("training", "fp32", "momentum", False): 12,  # 4 + 4 + 4
    ("training", "fp32", "adam", False): 16,  # 4 + 4 + 8
    ("training", "fp32", "adamw", False): 16,  # 4 + 4 + 8
    ("training", "fp16", "sgd", False): 4,  # 2 + 2 + 0
    ("training", "fp16", "sgd", True): 8,  # 2 + 2 + 4 + 0
    ("training", "fp16", "momentum", True): 12,  # 2 + 2 + 4 + 4
    ("training", "fp16", "adam", False): 12,  # 2 + 2 + 8
    ("training", "fp16", "adam", True): 16,  # 2 + 2 + 4 + 8
    ("training", "bf16", "adamw", True): 16,  # 2 + 2 + 4 + 8
    ("training", "bf16", "momentum", False): 8,  # 2 + 2 + 4
    ("training", "int8", "adam", False): 10,  # 1 + 1 + 8
    ("training", "fp8", "momentum", True): 10,  # 1 + 1 + 4 + 4
    ("training", "fp8", "sgd", False): 2,  # 1 + 1 + 0
}

N = 1000


def _call(**kwargs):
    try:
        return estimate_memory(**kwargs)
    except ValueError:
        pytest.fail("valid arguments were rejected: check the contract for this combination")
    except Exception as exc:
        pytest.fail(f"valid arguments crashed ({type(exc).__name__}): check the term formulas")


def _kwargs(mode, dtype, optimizer, master, *, activations=0, n=N):
    return {
        "num_params": n, "mode": mode, "dtype": dtype, "optimizer": optimizer,
        "master_weights": master, "activation_bytes": activations,
    }  # fmt: skip


@pytest.mark.parametrize("combo", list(BYTES_PER_PARAM), ids=lambda c: "-".join(map(str, c)))
def test_total_matches_the_hand_worked_bytes_per_parameter(combo):
    mode, dtype, optimizer, master = combo
    got = _call(**_kwargs(mode, dtype, optimizer, master))
    assert list(got) == list(TERMS), "return the six terms in the documented order"
    assert all(type(value) is int for value in got.values()), "every term must be a Python int"
    expected = BYTES_PER_PARAM[combo] * N
    if got["total"] != expected:
        pytest.fail(f"total bytes differ for this combination: {_diagnose(combo, got)}")


_STATES = {"none": 0, "sgd": 0, "momentum": 1, "adam": 2, "adamw": 2}
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}


def _diagnose(combo, got):
    """Name the most likely term mistake, given which term is off."""
    mode, dtype, optimizer, master = combo
    width, states = _DTYPE_BYTES[dtype], _STATES[optimizer]
    if mode == "training" and got["gradients"] == 0:
        return "gradient term missing: training keeps one gradient per weight"
    if (
        mode == "training"
        and states
        and width != 4
        and got["optimizer_states"] == states * N * width
    ):
        return "optimizer states stored in the working dtype: optimizer states are always fp32"
    if master and dtype != "fp32" and got["master_weights"] != 4 * N:
        return "master copy: four bytes per parameter when training with a lower-precision dtype"
    if not master and got["master_weights"]:
        return "master copy counted without master_weights: it is off unless requested"
    return "a term is off: check weights, gradients, master copy and optimizer states one by one"


def test_gradients_use_the_working_dtype_not_fp32():
    got = _call(**_kwargs("training", "bf16", "adam", True))
    assert got["gradients"] == 2 * N, (
        "gradients have the working dtype: two bytes per parameter for bf16, not four"
    )


def test_optimizer_states_are_always_fp32_even_for_low_precision_weights():
    got = _call(**_kwargs("training", "int8", "adam", False))
    assert got["optimizer_states"] == 8 * N, (
        "optimizer states are fp32: two states at four bytes each, whatever the weight dtype"
    )


def test_sixteen_byte_rule_holds_for_fp32_adam_as_well():
    got = _call(**_kwargs("training", "fp32", "adam", False))
    assert got["total"] == 16 * N, "fp32 Adam is 4 + 4 + 8 = 16 bytes per parameter"


def test_seven_billion_parameter_examples_are_exact():
    n = 7_000_000_000
    inf_bf16 = _call(**_kwargs("inference", "bf16", "none", False, n=n))
    inf_int8 = _call(**_kwargs("inference", "int8", "none", False, n=n))
    train = _call(**_kwargs("training", "bf16", "adamw", True, activations=2_000_000_000, n=n))
    if inf_bf16["total"] != 14_000_000_000:
        pytest.fail("7B bf16 inference must hold 14 GB of weights: two bytes per weight")
    if inf_int8["total"] != 7_000_000_000:
        pytest.fail("7B int8 inference must hold 7 GB of weights: one byte per weight")
    if train["total"] != 114_000_000_000:
        pytest.fail("7B mixed-precision AdamW training: 112 GB of state plus the activation term")


def test_activation_term_is_added_as_given():
    # A real activation size: one batch holding every image of the digits dataset as fp16,
    # two bytes for each of its 1,797 by 64 pixel values.
    rows, pixels = load("digits").X.shape
    held = 2 * int(rows) * int(pixels)
    got = _call(**_kwargs("training", "fp16", "momentum", True, activations=held))
    if got["activations"] != held:
        pytest.fail("the activation term must be the given number, unscaled")
    if got["total"] != 12 * N + held:
        pytest.fail("the total must include the activation term exactly once")


@pytest.mark.parametrize("optimizer", ["sgd"], ids=["sgd"])
def test_sgd_keeps_no_optimizer_state(optimizer):
    got = _call(**_kwargs("training", "fp32", optimizer, False))
    if got["optimizer_states"] != 0:
        pytest.fail("sgd without momentum keeps no optimizer state")


def test_zero_parameters_give_zero_bytes_except_activations():
    got = _call(**_kwargs("training", "bf16", "adam", True, activations=77, n=0))
    if got["total"] != 77 or got["weights"] != 0:
        pytest.fail("with zero parameters only the activation term may be non-zero")


def test_large_parameter_counts_stay_exact_integers():
    n = 10**30
    got = _call(**_kwargs("training", "bf16", "adam", True, n=n))
    if got["total"] != 16 * n:
        pytest.fail("large counts must use exact integers: keep the arithmetic in ints")


def _changed(**changes):
    def build():
        args = _kwargs("training", "bf16", "adam", True)
        args.update(changes)
        return args

    return build


INVALID_CASES = {
    "fp32-master": _changed(dtype="fp32", master_weights=True),
    "inference-master": _changed(mode="inference", optimizer="none", master_weights=True),
    "inference-adam": _changed(mode="inference", optimizer="adam", master_weights=False),
    "training-none": _changed(optimizer="none"),
    "unknown-dtype": _changed(dtype="float16"),
    "unknown-mode": _changed(mode="eval"),
    "unknown-optimizer": _changed(optimizer="lion"),
    "bool-params": _changed(num_params=True),
    "float-params": _changed(num_params=1000.0),
    "negative-params": _changed(num_params=-1),
    "negative-activations": _changed(activation_bytes=-5),
    "int-master": _changed(master_weights=1),
}


@pytest.mark.parametrize("case", list(INVALID_CASES), ids=list(INVALID_CASES))
def test_invalid_arguments_raise_value_error(case):
    args = INVALID_CASES[case]()
    try:
        estimate_memory(**args)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid argument ({case}) raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"invalid argument ({case}) was accepted instead of raising ValueError")
