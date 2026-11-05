"""Hidden tests: an independent computation over seeded random models and memory sizes, exact fit
and floor rules, weight rounding, KV-head divisibility, sliding windows, large exact counts, the
return type, config immutability, and validation.

Failure messages name the kind of mistake. They never print hidden inputs or expected values.
"""

from __future__ import annotations

import copy
import math
import random
from fractions import Fraction
from typing import Any

import pytest
from submission import min_gpu_count

GPU = 80 * 2**30
MHA8 = {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128}


def _oracle(  # noqa: PLR0917 (oracle takes the contract's arguments)
    params: int,
    cfg: dict[str, int],
    batch: int,
    context: int,
    width: int,
    memory: int,
    workspace: int,
    reserve: int,
) -> int:
    """Independent route: Fractions for the shares, and a degree list built from the KV heads."""
    usable = Fraction(memory * (100 - reserve), 100)
    window = cfg.get("sliding_window")
    stored = context if window is None else min(context, window)
    total_weights = Fraction(params * width)
    total_kv = Fraction(
        2 * cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"] * stored * batch * width
    )
    heads = cfg["num_kv_heads"]
    for exponent in range(64):
        degree = 2**exponent
        if degree > heads:
            break
        if heads % degree != 0:
            continue
        per_gpu = math.ceil(total_weights / degree) + total_kv / degree + workspace
        if per_gpu <= usable:
            return degree
    return 0


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_seeded_random_deployments_match_independent_computation() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic deployments, not security
    for _ in range(300):
        cfg = {
            "num_layers": rng.randint(1, 96),
            "num_kv_heads": rng.choice([1, 2, 4, 6, 8, 12, 16, 32, 64]),
            "head_dim": rng.choice([64, 128, 256]),
        }
        if rng.random() < 0.3:
            cfg["sliding_window"] = rng.randint(1, 8192)
        params = rng.randint(10**8, 2 * 10**11)
        batch = rng.randint(1, 256)
        context = rng.randint(1, 65536)
        width = rng.choice([1, 2, 4])
        memory = rng.randint(10**10, 2 * 10**11)
        workspace = rng.randint(0, 10**10)
        reserve = rng.randint(0, 99)
        want = _oracle(params, cfg, batch, context, width, memory, workspace, reserve)
        got = min_gpu_count(
            params,
            cfg,
            batch,
            context,
            width,
            memory,
            workspace_bytes=workspace,
            reserve_percent=reserve,
        )
        if type(got) is not int:
            pytest.fail("return a Python int degree, not a float")
        if got != want:
            pytest.fail(
                "the minimum degree disagrees with the independent computation: check the "
                "usable memory (floor after the reserve), the ceiling share of the weights, "
                "the KV split by KV heads, and that only power-of-two degrees dividing the "
                "KV heads are allowed"
            )


def test_total_exactly_equal_to_usable_memory_fits() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    # weights 9 bytes, KV 2 * 1 * 1 * 1 * 1 * 1 * 1 = 2 bytes at batch 1 and context 1, total 11.
    got = min_gpu_count(9, cfg, 1, 1, 1, 11, workspace_bytes=0, reserve_percent=0)
    if got != 1:
        pytest.fail("a total exactly equal to the usable memory must fit: compare with <=")


def test_one_byte_more_than_usable_memory_does_not_fit() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = min_gpu_count(9, cfg, 1, 1, 1, 10, workspace_bytes=0, reserve_percent=0)
    if got != 0:
        pytest.fail(
            "a total one byte above the usable memory must not fit: no degree is allowed to pass"
        )


def test_reserve_is_floored_after_the_percentage() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    # usable = 1001 * 90 // 100 = 900 (900.9 floored). A total of 900 fits, a total of 901 does not.
    fits = min_gpu_count(898, cfg, 1, 1, 1, 1001, workspace_bytes=0, reserve_percent=10)
    misses = min_gpu_count(899, cfg, 1, 1, 1, 1001, workspace_bytes=0, reserve_percent=10)
    if fits != 1 or misses != 0:
        pytest.fail("the usable memory must be floored after the reserve, not rounded up")


def test_reserve_floor_is_exact_at_large_memory_sizes() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    # usable = 693288195975 * 8 // 100 = 55463055678 exactly. A float percentage gives ...677,
    # one byte short, so a total of exactly 55463055678 (weights 55463055676 + KV 2) must fit.
    got = min_gpu_count(
        55463055676, cfg, 1, 1, 1, 693288195975, workspace_bytes=0, reserve_percent=92
    )
    if got != 1:
        pytest.fail(
            "the usable memory must be computed exactly with integers, not as a float percentage"
        )


def test_weight_shard_is_rounded_up_when_it_does_not_divide() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 2, "head_dim": 1}
    # weights 3 bytes split over 2 GPUs: each holds ceil(3 / 2) = 2 bytes. KV at degree 2 is
    # 2 bytes.
    # Total per GPU is 4, so a memory of 3 must not fit at any degree.
    got = min_gpu_count(3, cfg, 1, 1, 1, 3, workspace_bytes=0, reserve_percent=0)
    if got != 0:
        pytest.fail(
            "each GPU's weight share is rounded up, so a 4-byte need must not fit in 3 bytes"
        )


def test_degree_that_does_not_divide_kv_heads_is_never_returned() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 6, "head_dim": 1}
    # Degree 2 needs 506 bytes per GPU and degree 4 (which 6 KV heads cannot split) would need 253.
    # The allowed degrees are 1 and 2 only, so with 300 bytes the answer is 0.
    got = min_gpu_count(1000, cfg, 1, 1, 1, 300, workspace_bytes=0, reserve_percent=0)
    if got != 0:
        pytest.fail(
            "a degree that does not divide the KV heads is not allowed: with 6 KV heads, "
            "only 1 and 2 are candidates"
        )


def test_sliding_window_caps_the_cached_tokens() -> None:
    cfg = {"num_layers": 1, "num_kv_heads": 2, "head_dim": 1, "sliding_window": 1}
    # With the window, one token is stored: KV 2 * 1 * 2 * 1 * 1 = 4 bytes at degree 1. Without the
    # window, context 100 would need 400 bytes. A memory of 10 bytes must fit at degree 1.
    got = min_gpu_count(1, cfg, 1, 100, 1, 10, workspace_bytes=0, reserve_percent=0)
    if got != 1:
        pytest.fail(
            "a sliding window caps the stored tokens at the window, so the cache stops growing"
        )


def test_sliding_window_cap_is_exact_at_the_window_boundary() -> None:
    # Weights 1 byte, KV 2 bytes per stored token: window 50 needs 1 + 100 = 101 bytes exactly, so
    # the window must be used as stored tokens and not off by one (51 tokens would need 103).
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1, "sliding_window": 50}
    got = min_gpu_count(1, cfg, 1, 4096, 1, 101, workspace_bytes=0, reserve_percent=0)
    if got != 1:
        pytest.fail(
            "the window is the exact stored-token count: a cap one token too large misses the fit"
        )


def test_sliding_window_of_wrong_type_raises_type_error() -> None:
    cfg: dict[str, Any] = {
        "num_layers": 1,
        "num_kv_heads": 1,
        "head_dim": 1,
        "sliding_window": "64",
    }
    _expect_error(
        TypeError,
        lambda: min_gpu_count(1, cfg, 1, 8, 1, 10, workspace_bytes=0, reserve_percent=0),
        "sliding_window must be None, missing, or a Python int: other types raise TypeError",
    )


def test_weights_that_never_fit_return_zero() -> None:
    got = min_gpu_count(
        10**12,
        {"num_layers": 1, "num_kv_heads": 8, "head_dim": 1},
        1,
        1,
        2,
        GPU,
        workspace_bytes=0,
        reserve_percent=0,
    )
    if got != 0:
        pytest.fail("when no allowed degree holds the weights, return 0")


def test_large_byte_counts_stay_exact() -> None:
    cfg = {"num_layers": 96, "num_kv_heads": 8, "head_dim": 256}
    memory = 2**60 + 1
    params = 2**58
    got = min_gpu_count(params, cfg, 2**20, 2**16, 2, memory, workspace_bytes=0, reserve_percent=0)
    want = _oracle(params, cfg, 2**20, 2**16, 2, memory, 0, 0)
    if got != want:
        pytest.fail(
            "byte counts beyond float precision must stay exact: use integers, not float division"
        )


def test_zero_workspace_and_zero_reserve_are_allowed() -> None:
    got = min_gpu_count(7 * 10**9, MHA8, 1, 4096, 2, GPU, workspace_bytes=0, reserve_percent=0)
    if type(got) is not int or got not in {1, 2, 4, 8}:
        pytest.fail(
            "with no workspace and no reserve the answer must be a Python int allowed degree"
        )


def test_config_is_not_mutated() -> None:
    cfg = dict(MHA8, sliding_window=4096, hidden_size=8192)
    before = copy.deepcopy(cfg)
    min_gpu_count(70 * 10**9, cfg, 4, 4096, 2, GPU, workspace_bytes=0, reserve_percent=10)
    if cfg != before:
        pytest.fail("the config dict was changed: read the keys without writing to the input")


@pytest.mark.parametrize(
    ("params", "cfg", "batch", "context", "width", "memory", "workspace", "reserve", "bad"),
    [
        (True, MHA8, 1, 1, 2, GPU, 0, 0, "params"),
        (70 * 10**9, MHA8, 1.0, 1, 2, GPU, 0, 0, "batch"),
        (70 * 10**9, MHA8, 1, "4096", 2, GPU, 0, 0, "context_len"),
        (70 * 10**9, MHA8, 1, 1, None, GPU, 0, 0, "element_bytes"),
        (70 * 10**9, MHA8, 1, 1, 2, 2.0**36, 0, 0, "gpu_memory_bytes"),
        (70 * 10**9, MHA8, 1, 1, 2, GPU, True, 0, "workspace_bytes"),
        (70 * 10**9, MHA8, 1, 1, 2, GPU, 0, 10.0, "reserve_percent"),
        (
            70 * 10**9,
            {"num_layers": True, "num_kv_heads": 8, "head_dim": 128},
            1,
            1,
            2,
            GPU,
            0,
            0,
            "num_layers",
        ),
    ],
    ids=[
        "bool-params",
        "float-batch",
        "string-context",
        "none-element-bytes",
        "float-memory",
        "bool-workspace",
        "float-reserve",
        "bool-layers",
    ],
)
def test_non_int_value_raises_type_error(  # noqa: PLR0917 (parametrized arguments)
    params: Any,
    cfg: Any,
    batch: Any,
    context: Any,
    width: Any,
    memory: Any,
    workspace: Any,
    reserve: Any,
    bad: str,
) -> None:
    _expect_error(
        TypeError,
        lambda: min_gpu_count(
            params,
            cfg,
            batch,
            context,
            width,
            memory,
            workspace_bytes=workspace,
            reserve_percent=reserve,
        ),
        f"{bad} must be a Python int: bool, float, str and None are rejected",
    )


def test_config_that_is_not_a_dict_raises_type_error() -> None:
    not_a_dict: Any = ["num_layers"]
    _expect_error(
        TypeError,
        lambda: min_gpu_count(
            10**9,
            not_a_dict,
            1,
            1,
            2,
            GPU,
            workspace_bytes=0,
            reserve_percent=0,
        ),
        "config must be a dict",
    )


@pytest.mark.parametrize(
    ("params", "cfg", "batch", "context", "width", "memory", "workspace", "reserve"),
    [
        (0, MHA8, 1, 1, 2, GPU, 0, 0),
        (70 * 10**9, MHA8, 0, 1, 2, GPU, 0, 0),
        (70 * 10**9, MHA8, 1, 0, 2, GPU, 0, 0),
        (70 * 10**9, MHA8, 1, 1, 0, GPU, 0, 0),
        (70 * 10**9, MHA8, 1, 1, 2, 0, 0, 0),
        (70 * 10**9, MHA8, 1, 1, 2, GPU, -1, 0),
        (70 * 10**9, MHA8, 1, 1, 2, GPU, 0, -1),
        (70 * 10**9, MHA8, 1, 1, 2, GPU, 0, 100),
        (70 * 10**9, {"num_layers": 80, "num_kv_heads": 8}, 1, 1, 2, GPU, 0, 0),
        (70 * 10**9, dict(MHA8, sliding_window=0), 1, 1, 2, GPU, 0, 0),
    ],
    ids=[
        "zero-params",
        "zero-batch",
        "zero-context",
        "zero-element-bytes",
        "zero-memory",
        "negative-workspace",
        "negative-reserve",
        "reserve-100",
        "missing-head-dim",
        "zero-window",
    ],
)
def test_out_of_range_or_missing_value_raises_value_error(  # noqa: PLR0917
    params: int,
    cfg: dict[str, int],
    batch: int,
    context: int,
    width: int,
    memory: int,
    workspace: int,
    reserve: int,
) -> None:
    _expect_error(
        ValueError,
        lambda: min_gpu_count(
            params,
            cfg,
            batch,
            context,
            width,
            memory,
            workspace_bytes=workspace,
            reserve_percent=reserve,
        ),
        "a value out of range, a missing key, or a reserve of 100 or more must raise ValueError",
    )


def test_zero_head_dim_raises_value_error() -> None:
    # The contract says each of num_layers, num_kv_heads and head_dim is at least 1.
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 0}
    _expect_error(
        ValueError,
        lambda: min_gpu_count(1, cfg, 1, 1, 1, 10, workspace_bytes=0, reserve_percent=0),
        "a head_dim of zero is out of range and must raise ValueError",
    )


def test_zero_kv_heads_raises_value_error() -> None:
    # The contract says each of num_layers, num_kv_heads and head_dim is at least 1.
    cfg = {"num_layers": 1, "num_kv_heads": 0, "head_dim": 1}
    _expect_error(
        ValueError,
        lambda: min_gpu_count(1, cfg, 1, 1, 1, 10, workspace_bytes=0, reserve_percent=0),
        "a num_kv_heads of zero is out of range and must raise ValueError",
    )


def test_weight_shard_is_rounded_up_beyond_float_precision() -> None:
    # Weights 2**60 + 1 split over 2 GPUs: each holds ceil = 2**59 + 1 bytes. KV at degree 2 is
    # 2 bytes, so the total is 2**59 + 3, which does not fit in 2**59 + 2.
    cfg = {"num_layers": 1, "num_kv_heads": 2, "head_dim": 1}
    got = min_gpu_count(2**60 + 1, cfg, 1, 1, 1, 2**59 + 2, workspace_bytes=0, reserve_percent=0)
    if got != 0:
        pytest.fail(
            "the weight shard must be rounded up with integer arithmetic, not a float share"
        )


def test_usable_memory_is_floored_exactly_beyond_float_precision() -> None:
    # With no reserve, usable is the full memory. A total of exactly 2**60 + 1 must fit.
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = min_gpu_count(2**60 - 1, cfg, 1, 1, 1, 2**60 + 1, workspace_bytes=0, reserve_percent=0)
    if got != 1:
        pytest.fail(
            "a total equal to the usable memory must fit even beyond float precision: "
            "use integer arithmetic for the usable memory"
        )


def test_kv_bytes_stay_exact_beyond_float_precision() -> None:
    # One GPU: weights 1 byte, KV 2 * (2**59 + 1) = 2**60 + 2 bytes, total 2**60 + 3, which does
    # not fit in 2**60 + 2.
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = min_gpu_count(1, cfg, 1, 2**59 + 1, 1, 2**60 + 2, workspace_bytes=0, reserve_percent=0)
    if got != 0:
        pytest.fail(
            "KV bytes beyond float precision must stay exact: use integer arithmetic for the cache"
        )
