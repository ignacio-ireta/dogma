"""Hidden tests: an exact rational oracle on seeded random models and devices, the weight
sharing and monotonicity properties, the byte-count rules (KV factor, context plus new
token), and validation.

Failure messages name the kind of mistake. They never print hidden inputs or expected values.
"""

from __future__ import annotations

import copy
import math
import random
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import pytest
from submission import serving_step_times

PEAK = 312_000_000_000_000
BW = 2_000_000_000_000
MHA = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
GQA = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
KEYS = {"prefill_s", "decode_step_s", "decode_tokens_per_s"}


def _exact(  # noqa: PLR0917 (oracle takes the contract's arguments)
    params: int,
    cfg: dict[str, int],
    prompt: int,
    context: int,
    batch: int,
    width: int,
    peak: int,
    bw: int,
) -> dict[str, Fraction]:
    weights = params * width
    kv = 2 * cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"] * width
    prefill = max(
        Fraction(2 * params * prompt * batch, peak), Fraction(weights + batch * prompt * kv, bw)
    )
    step = max(
        Fraction(2 * params * batch, peak), Fraction(weights + batch * (context + 1) * kv, bw)
    )
    return {
        "prefill_s": prefill,
        "decode_step_s": step,
        "decode_tokens_per_s": Fraction(batch) / step,
    }


def _same_float(got: Any, expected: Fraction, mistake: str) -> None:
    if type(got) is not float:
        pytest.fail(f"{mistake}; the value must be a float, not {type(got).__name__}")
    if not math.isclose(got, float(expected), rel_tol=1e-9, abs_tol=0.0):
        pytest.fail(mistake)


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_seeded_random_models_match_exact_rational_model() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic models, not security
    for _ in range(300):
        params = rng.randint(10**6, 10**11)
        cfg = {
            "num_layers": rng.randint(1, 96),
            "num_kv_heads": rng.randint(1, 64),
            "head_dim": rng.choice([32, 64, 128, 256]),
        }
        prompt = rng.randint(1, 8192)
        context = rng.randint(0, 8192)
        batch = rng.randint(1, 64)
        width = rng.choice([1, 2, 4])
        peak = rng.randint(10**12, 10**15)
        bw = rng.randint(10**10, 10**13)
        want = _exact(params, cfg, prompt, context, batch, width, peak, bw)
        got = serving_step_times(
            params, cfg, prompt, context, batch, width, peak_flops=peak, bandwidth=bw
        )
        _same_float(
            got["prefill_s"],
            want["prefill_s"],
            "prefill time disagrees: FLOPs are 2 * params * prompt * batch; bytes are the "
            "weights plus batch * prompt * KV bytes per token",
        )
        _same_float(
            got["decode_step_s"],
            want["decode_step_s"],
            "decode step disagrees: bytes are the weights plus batch * (context + 1) * KV "
            "per token, and FLOPs are 2 * params * batch",
        )
        _same_float(
            got["decode_tokens_per_s"],
            want["decode_tokens_per_s"],
            "decode throughput disagrees: it is batch divided by the decode step time",
        )


def test_decode_reads_the_weights_once_per_step_not_per_sequence() -> None:
    one = serving_step_times(10**9, GQA, 64, 64, 1, 2, peak_flops=PEAK, bandwidth=BW)
    eight = serving_step_times(10**9, GQA, 64, 64, 8, 2, peak_flops=PEAK, bandwidth=BW)
    if not eight["decode_step_s"] < 8 * one["decode_step_s"]:
        pytest.fail(
            "a decode step of 8 sequences must cost less than 8 single-sequence steps: "
            "the weights are read once per step, not once per sequence"
        )


def test_decode_throughput_never_falls_as_the_batch_grows() -> None:
    previous = 0.0
    for batch in range(1, 65):
        rate = serving_step_times(
            7 * 10**9, MHA, 512, 512, batch, 2, peak_flops=PEAK, bandwidth=BW
        )["decode_tokens_per_s"]
        if rate < previous * (1 - 1e-12):
            pytest.fail(
                "decode throughput fell as the batch grew: the batch shares the weight "
                "read, so tokens per second cannot drop"
            )
        previous = rate


def test_kv_factor_of_two_counts_keys_and_values_in_decode_bytes() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = serving_step_times(1, cfg, 1, 9, 3, 1, peak_flops=10**18, bandwidth=1)
    # bytes = weights 1 + batch 3 * (context 9 + 1) * kv 2 = 61, with bandwidth 1 byte/s.
    _same_float(
        got["decode_step_s"],
        Fraction(61),
        "decode bytes are wrong: keys and values are both read, so the KV term has a factor of 2",
    )


def test_decode_reads_the_cached_tokens_plus_the_new_one() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = serving_step_times(1, cfg, 1, 0, 1, 1, peak_flops=10**18, bandwidth=1)
    _same_float(
        got["decode_step_s"],
        Fraction(3),
        "with an empty cache the decode step still reads the new token: 1 weight byte plus 2",
    )


def test_prefill_flops_scale_with_both_prompt_length_and_batch() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = serving_step_times(5, cfg, 7, 0, 3, 1, peak_flops=1, bandwidth=10**18)
    _same_float(
        got["prefill_s"],
        Fraction(2 * 5 * 7 * 3),
        "prefill FLOPs are 2 * params * prompt * batch: a compute-bound prefill scales with both",
    )


def test_single_token_prefill_is_memory_bound_on_the_weight_read() -> None:
    got = serving_step_times(7 * 10**9, MHA, 1, 0, 1, 2, peak_flops=PEAK, bandwidth=BW)
    _same_float(
        got["prefill_s"],
        Fraction(7 * 10**9 * 2 + 524_288, BW),
        "a one-token prefill is memory-bound: weight bytes plus one token of KV, over bandwidth",
    )


def test_config_is_not_mutated_and_extra_keys_are_ignored() -> None:
    cfg = dict(GQA, sliding_window=4096, hidden_size=4096)
    before = copy.deepcopy(cfg)
    serving_step_times(10**9, cfg, 64, 64, 2, 2, peak_flops=PEAK, bandwidth=BW)
    if cfg != before:
        pytest.fail("the config dict was changed: read the keys without writing to the input")


def test_returns_exact_key_set_and_float_values() -> None:
    got = serving_step_times(10**9, GQA, 64, 64, 2, 2, peak_flops=PEAK, bandwidth=BW)
    if set(got) != KEYS:
        pytest.fail(
            "the result must have exactly the keys prefill_s, decode_step_s, decode_tokens_per_s"
        )
    for key in KEYS:
        if type(got[key]) is not float:
            pytest.fail(f"{key} must be a float number of seconds or tokens per second")


@pytest.mark.parametrize(
    ("params", "cfg", "prompt", "context", "batch", "width", "peak", "bw", "bad"),
    [
        (True, MHA, 4, 4, 1, 2, PEAK, BW, "params"),
        (7 * 10**9, MHA, 4.0, 4, 1, 2, PEAK, BW, "prompt_len"),
        (7 * 10**9, MHA, 4, 4, "2", 2, PEAK, BW, "batch"),
        (7 * 10**9, MHA, 4, 4, 1, None, PEAK, BW, "element_bytes"),
        (7 * 10**9, MHA, 4, 4, 1, 2, 1.5, BW, "peak_flops"),
        (7 * 10**9, MHA, 4, 4, 1, 2, PEAK, True, "bandwidth"),
        (
            7 * 10**9,
            {"num_layers": True, "num_kv_heads": 1, "head_dim": 1},
            4,
            4,
            1,
            2,
            PEAK,
            BW,
            "num_layers",
        ),
        (7 * 10**9, MHA, 4, 4.5, 1, 2, PEAK, BW, "context_len"),
    ],
    ids=[
        "bool-params",
        "float-prompt",
        "string-batch",
        "none-element-bytes",
        "float-peak",
        "bool-bandwidth",
        "bool-layers",
        "float-context",
    ],
)
def test_non_int_value_raises_type_error(  # noqa: PLR0917 (parametrized arguments)
    params: Any,
    cfg: Any,
    prompt: Any,
    context: Any,
    batch: Any,
    width: Any,
    peak: Any,
    bw: Any,
    bad: str,
) -> None:
    _expect_error(
        TypeError,
        lambda: serving_step_times(
            params, cfg, prompt, context, batch, width, peak_flops=peak, bandwidth=bw
        ),
        f"{bad} must be a Python int: bool, float, str and None are rejected",
    )


def test_config_that_is_not_a_dict_raises_type_error() -> None:
    not_a_dict: Any = ["num_layers"]
    _expect_error(
        TypeError,
        lambda: serving_step_times(
            10**9,
            not_a_dict,
            4,
            4,
            1,
            2,
            peak_flops=PEAK,
            bandwidth=BW,
        ),
        "config must be a dict",
    )


def test_config_mapping_that_is_not_a_dict_raises_type_error() -> None:
    # A read-only mapping supports `in` and indexing, so only an explicit dict check rejects it.
    not_a_dict: Any = MappingProxyType(dict(MHA))
    _expect_error(
        TypeError,
        lambda: serving_step_times(
            10**9,
            not_a_dict,
            4,
            4,
            1,
            2,
            peak_flops=PEAK,
            bandwidth=BW,
        ),
        "config must be a dict: a mapping that is not a dict must raise TypeError",
    )


@pytest.mark.parametrize(
    ("params", "cfg", "prompt", "context", "batch", "width", "peak", "bw"),
    [
        (0, MHA, 4, 4, 1, 2, PEAK, BW),
        (7 * 10**9, MHA, 0, 4, 1, 2, PEAK, BW),
        (7 * 10**9, MHA, 4, -1, 1, 2, PEAK, BW),
        (7 * 10**9, MHA, 4, 4, 0, 2, PEAK, BW),
        (7 * 10**9, MHA, 4, 4, 1, 0, PEAK, BW),
        (7 * 10**9, MHA, 4, 4, 1, 2, 0, BW),
        (7 * 10**9, MHA, 4, 4, 1, 2, PEAK, 0),
        (7 * 10**9, {"num_layers": 32, "num_kv_heads": 32}, 4, 4, 1, 2, PEAK, BW),
    ],
    ids=[
        "zero-params",
        "empty-prompt",
        "negative-context",
        "zero-batch",
        "zero-element-bytes",
        "zero-peak",
        "zero-bandwidth",
        "missing-head-dim",
    ],
)
def test_value_below_minimum_or_missing_key_raises_value_error(  # noqa: PLR0917
    params: int,
    cfg: dict[str, int],
    prompt: int,
    context: int,
    batch: int,
    width: int,
    peak: int,
    bw: int,
) -> None:
    _expect_error(
        ValueError,
        lambda: serving_step_times(
            params, cfg, prompt, context, batch, width, peak_flops=peak, bandwidth=bw
        ),
        "a value below its minimum, or a missing config key, must raise ValueError",
    )


def test_zero_num_layers_raises_value_error() -> None:
    # The contract says num_layers is an int of at least 1, so a zero is out of range.
    cfg = {"num_layers": 0, "num_kv_heads": 32, "head_dim": 128}
    _expect_error(
        ValueError,
        lambda: serving_step_times(7 * 10**9, cfg, 4, 4, 1, 2, peak_flops=PEAK, bandwidth=BW),
        "a num_layers of zero is out of range and must raise ValueError",
    )


def test_zero_num_kv_heads_raises_value_error() -> None:
    # The contract says num_kv_heads is an int of at least 1, so a zero is out of range.
    cfg = {"num_layers": 32, "num_kv_heads": 0, "head_dim": 128}
    _expect_error(
        ValueError,
        lambda: serving_step_times(7 * 10**9, cfg, 4, 4, 1, 2, peak_flops=PEAK, bandwidth=BW),
        "a num_kv_heads of zero is out of range and must raise ValueError",
    )


def test_zero_head_dim_raises_value_error() -> None:
    # The contract says head_dim is an int of at least 1, so a zero is out of range.
    cfg = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 0}
    _expect_error(
        ValueError,
        lambda: serving_step_times(7 * 10**9, cfg, 4, 4, 1, 2, peak_flops=PEAK, bandwidth=BW),
        "a head_dim of zero is out of range and must raise ValueError",
    )
