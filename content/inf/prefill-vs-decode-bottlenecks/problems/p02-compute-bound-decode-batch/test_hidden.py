"""Hidden tests: an independent search over batch sizes on seeded random models and devices,
the smallest-batch property, the tie rule, exact integers beyond float precision, the return
type, config immutability, and validation.

Failure messages name the kind of mistake. They never print hidden inputs or expected values.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest
from submission import compute_bound_decode_batch

PEAK = 312_000_000_000_000
BW = 2_000_000_000_000
GQA = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128}
MHA = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
UPPER = 2**100  # far above any threshold the random ranges can produce


def _compute_bound(  # noqa: PLR0917 (oracle takes the contract's arguments)
    params: int, cfg: dict[str, int], context: int, width: int, peak: int, bw: int, batch: int
) -> bool:
    """Is one decode step at this batch compute-bound? Exact integers, the tie counts as compute."""
    weights = params * width
    kv = 2 * cfg["num_layers"] * cfg["num_kv_heads"] * cfg["head_dim"] * width
    flops = 2 * params * batch
    traffic = weights + batch * (context + 1) * kv
    return flops * bw >= peak * traffic


def _search_oracle(  # noqa: PLR0917 (oracle takes the contract's arguments)
    params: int, cfg: dict[str, int], context: int, width: int, peak: int, bw: int
) -> int:
    """Independent answer: bisect over batch sizes using the exact predicate; 0 if none in range."""

    def ok(batch: int) -> bool:
        return _compute_bound(params, cfg, context, width, peak, bw, batch)

    if not ok(UPPER):
        return 0
    if ok(1):
        return 1
    low, high = 1, UPPER  # ok(low) is False, ok(high) is True
    while high - low > 1:
        mid = (low + high) // 2
        if ok(mid):
            high = mid
        else:
            low = mid
    return high


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_seeded_random_models_match_an_independent_search_over_batches() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic models, not security
    for _ in range(300):
        params = rng.randint(10**6, 10**11)
        cfg = {
            "num_layers": rng.randint(1, 96),
            "num_kv_heads": rng.randint(1, 64),
            "head_dim": rng.choice([64, 128, 256]),
        }
        context = rng.randint(0, 8192)
        width = rng.choice([1, 2, 4])
        peak = rng.randint(10**12, 10**15)
        bw = rng.randint(10**10, 10**13)
        want = _search_oracle(params, cfg, context, width, peak, bw)
        got = compute_bound_decode_batch(params, cfg, context, width, peak_flops=peak, bandwidth=bw)
        if type(got) is not int:
            pytest.fail("return a Python int batch size, not a float")
        if got != want:
            pytest.fail(
                "the batch threshold disagrees with the search: solve "
                "2 * params * B * bandwidth >= peak * (weights + B * (context + 1) * kv) for the "
                "smallest whole B, and return 0 when the slope is not positive"
            )


def test_returned_batch_is_compute_bound_and_the_one_before_it_is_not() -> None:
    got = compute_bound_decode_batch(7 * 10**9, GQA, 512, 2, peak_flops=PEAK, bandwidth=BW)
    if not _compute_bound(7 * 10**9, GQA, 512, 2, PEAK, BW, got):
        pytest.fail("the returned batch is not compute-bound: the threshold is too small")
    if got > 1 and _compute_bound(7 * 10**9, GQA, 512, 2, PEAK, BW, got - 1):
        pytest.fail(
            "the batch before the returned one is already compute-bound: the threshold is too large"
        )


def test_intensity_exactly_at_the_ridge_counts_as_compute_bound() -> None:
    # params 2, fp8-width 1, one layer, one head, head_dim 1, no context: at batch 1 the step has
    # 4 FLOPs and 4 bytes, and the ridge is 1 FLOP/byte, so the step sits exactly on the ridge.
    tie_model = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    got = compute_bound_decode_batch(2, tie_model, 0, 1, peak_flops=1, bandwidth=1)
    if got != 1:
        pytest.fail(
            "a decode step exactly at the ridge must count as compute-bound, so the answer is 1: "
            "use >= in the compute test"
        )


def test_threshold_beyond_float_precision_is_an_exact_ceiling() -> None:
    # The threshold is (2**71 + 1) / 2, which is 2**70 + 0.5. A float division rounds it down to
    # 2**70, one batch short. The exact ceiling is 2**70 + 1.
    big_model = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 2**71}
    got = compute_bound_decode_batch(2**71 + 1, big_model, 0, 1, peak_flops=1, bandwidth=1)
    if got != 2**70 + 1:
        pytest.fail(
            "the threshold is off by one for a very large batch: take the ceiling with integers, "
            "not a float division"
        )


def test_model_with_kv_traffic_above_the_flops_never_reaches_the_ridge() -> None:
    got = compute_bound_decode_batch(7 * 10**9, MHA, 8192, 2, peak_flops=PEAK, bandwidth=BW)
    if got != 0:
        pytest.fail(
            "with KV traffic that grows faster than FLOPs, no batch is compute-bound: return 0"
        )


def test_asymptotic_intensity_exactly_equal_to_ridge_is_never_reached() -> None:
    # Intensity tends to 2P/((c+1)*kv) = 2 from below, so the ridge (1) is only a limit.
    # Every finite batch keeps the weight bytes, so the step stays memory-bound: the answer is 0.
    tiny = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}
    try:
        got = compute_bound_decode_batch(1, tiny, 0, 1, peak_flops=1, bandwidth=1)
    except ZeroDivisionError:
        pytest.fail("a zero slope divides by zero: test the sign of the slope before dividing")
    if got != 0:
        pytest.fail(
            "a limit that only equals the ridge is never reached at a finite batch: return 0"
        )


def test_result_is_an_int_and_never_negative() -> None:
    got = compute_bound_decode_batch(7 * 10**9, GQA, 0, 2, peak_flops=PEAK, bandwidth=BW)
    if type(got) is not int or got < 0:
        pytest.fail("return a non-negative Python int batch size")


def test_config_is_not_mutated() -> None:
    cfg = dict(GQA, sliding_window=4096, hidden_size=4096)
    before = copy.deepcopy(cfg)
    compute_bound_decode_batch(7 * 10**9, cfg, 512, 2, peak_flops=PEAK, bandwidth=BW)
    if cfg != before:
        pytest.fail("the config dict was changed: read the keys without writing to the input")


@pytest.mark.parametrize(
    ("params", "cfg", "context", "width", "peak", "bw", "bad"),
    [
        (True, GQA, 512, 2, PEAK, BW, "params"),
        (7 * 10**9, GQA, 512.0, 2, PEAK, BW, "context_len"),
        (7 * 10**9, GQA, 512, None, PEAK, BW, "element_bytes"),
        (7 * 10**9, GQA, 512, 2, PEAK, "2e12", "bandwidth"),
        (
            7 * 10**9,
            {"num_layers": True, "num_kv_heads": 8, "head_dim": 128},
            512,
            2,
            PEAK,
            BW,
            "num_layers",
        ),
    ],
    ids=["bool-params", "float-context", "none-element-bytes", "string-bandwidth", "bool-layers"],
)
def test_non_int_value_raises_type_error(  # noqa: PLR0917 (parametrized arguments)
    params: Any, cfg: Any, context: Any, width: Any, peak: Any, bw: Any, bad: str
) -> None:
    _expect_error(
        TypeError,
        lambda: compute_bound_decode_batch(
            params, cfg, context, width, peak_flops=peak, bandwidth=bw
        ),
        f"{bad} must be a Python int: bool, float, str and None are rejected",
    )


def test_config_that_is_not_a_dict_raises_type_error() -> None:
    not_a_dict: Any = ["num_layers"]
    _expect_error(
        TypeError,
        lambda: compute_bound_decode_batch(
            10**9,
            not_a_dict,
            4,
            2,
            peak_flops=PEAK,
            bandwidth=BW,
        ),
        "config must be a dict",
    )


@pytest.mark.parametrize(
    ("params", "cfg", "context", "width", "peak", "bw"),
    [
        (0, GQA, 512, 2, PEAK, BW),
        (7 * 10**9, GQA, -1, 2, PEAK, BW),
        (7 * 10**9, GQA, 512, 0, PEAK, BW),
        (7 * 10**9, GQA, 512, 2, 0, BW),
        (7 * 10**9, GQA, 512, 2, PEAK, 0),
        (7 * 10**9, {"num_layers": 32, "num_kv_heads": 8}, 512, 2, PEAK, BW),
    ],
    ids=[
        "zero-params",
        "negative-context",
        "zero-element-bytes",
        "zero-peak",
        "zero-bandwidth",
        "missing-head-dim",
    ],
)
def test_value_below_minimum_or_missing_key_raises_value_error(  # noqa: PLR0917
    params: int, cfg: dict[str, int], context: int, width: int, peak: int, bw: int
) -> None:
    _expect_error(
        ValueError,
        lambda: compute_bound_decode_batch(
            params, cfg, context, width, peak_flops=peak, bandwidth=bw
        ),
        "a value below its minimum, or a missing config key, must raise ValueError",
    )


def test_zero_num_layers_raises_value_error() -> None:
    # The contract says num_layers is an int of at least 1, so a zero is out of range.
    cfg = {"num_layers": 0, "num_kv_heads": 8, "head_dim": 128}
    _expect_error(
        ValueError,
        lambda: compute_bound_decode_batch(7 * 10**9, cfg, 512, 2, peak_flops=PEAK, bandwidth=BW),
        "a num_layers of zero is out of range and must raise ValueError",
    )


def test_zero_num_kv_heads_raises_value_error() -> None:
    # The contract says num_kv_heads is an int of at least 1, so a zero is out of range.
    cfg = {"num_layers": 32, "num_kv_heads": 0, "head_dim": 128}
    _expect_error(
        ValueError,
        lambda: compute_bound_decode_batch(7 * 10**9, cfg, 512, 2, peak_flops=PEAK, bandwidth=BW),
        "a num_kv_heads of zero is out of range and must raise ValueError",
    )


def test_zero_head_dim_raises_value_error() -> None:
    # The contract says head_dim is an int of at least 1, so a zero is out of range.
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 0}
    _expect_error(
        ValueError,
        lambda: compute_bound_decode_batch(7 * 10**9, cfg, 512, 2, peak_flops=PEAK, bandwidth=BW),
        "a head_dim of zero is out of range and must raise ValueError",
    )


def test_sliding_window_key_does_not_change_the_threshold() -> None:
    # Other keys are ignored, including sliding_window: attention is treated as full.
    plain = compute_bound_decode_batch(7_000_000_000, GQA, 512, 2, peak_flops=PEAK, bandwidth=BW)
    windowed = dict(GQA, sliding_window=64)
    got = compute_bound_decode_batch(7_000_000_000, windowed, 512, 2, peak_flops=PEAK, bandwidth=BW)
    assert got == plain, (
        "the answer depends on sliding_window: the key must be ignored, attention is full"
    )
