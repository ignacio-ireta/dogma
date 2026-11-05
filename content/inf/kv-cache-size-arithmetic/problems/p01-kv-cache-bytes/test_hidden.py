"""Hidden tests: an independent formula on seeded random configs, plus edge and validation rules.

Failure messages name the kind of mistake. They never print hidden configs or expected values.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest
from submission import kv_cache_bytes

BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def _same(got: Any, expected: int, mistake: str) -> None:
    if type(got) is not int:
        pytest.fail("return a Python int of bytes, not a float or another numeric type")
    if got != expected:
        pytest.fail(mistake)


def _oracle(config: dict[str, Any], seq_len: int, batch: int, dtype: str) -> int:
    """A separate expression of the rule: a product of factors, with the window as a min."""
    window = config.get("sliding_window")
    stored = seq_len if window is None else min(seq_len, window)
    factors = (2, config["num_layers"], config["num_kv_heads"], config["head_dim"], stored, batch)
    return math.prod(factors) * BYTES[dtype]


def _expect_error(exc_type: type[Exception], call: Any, mistake: str) -> None:
    try:
        call()
    except exc_type:
        return
    except Exception as exc:
        pytest.fail(f"{mistake}; instead it raised {type(exc).__name__}")
    pytest.fail(mistake)


def test_seeded_random_configs_match_independent_formula() -> None:
    rng = random.Random(2026)  # noqa: S311 -- seeded synthetic configs, not security.
    dtypes = sorted(BYTES)
    for _ in range(300):
        config: dict[str, Any] = {
            "num_layers": rng.randint(1, 96),
            "num_kv_heads": rng.randint(1, 64),
            "head_dim": rng.choice([32, 64, 96, 128, 256]),
        }
        if rng.random() < 0.5:
            config["sliding_window"] = rng.randint(1, 8192)
        seq_len = rng.randint(0, 20000)
        batch = rng.randint(1, 16)
        dtype = rng.choice(dtypes)
        expected = _oracle(config, seq_len, batch, dtype)
        _same(
            kv_cache_bytes(config, seq_len, batch, dtype),
            expected,
            "the count disagrees with the formula: check the K and V factor, the KV-head count, "
            "the window cap min(context, window), and the bytes per element for the dtype",
        )


def test_grouped_query_cache_scales_with_kv_heads_not_query_heads() -> None:
    base = {"num_layers": 32, "head_dim": 128, "num_kv_heads": 8, "num_attention_heads": 32}
    wide = dict(base, num_kv_heads=32, num_attention_heads=32)
    _same(
        kv_cache_bytes(base, 4096),
        kv_cache_bytes(wide, 4096) // 4,
        "the cache must scale with the KV-head count: it is quartered when 8 KV heads replace 32",
    )


def test_multi_query_attention_keeps_one_kv_head() -> None:
    mqa = {"num_layers": 40, "num_kv_heads": 1, "head_dim": 128}
    _same(
        kv_cache_bytes(mqa, 1000),
        2 * 40 * 1 * 128 * 1000 * 1 * 2,
        "multi-query attention stores one KV head: the count of KV heads is wrong",
    )


def test_window_larger_than_context_changes_nothing() -> None:
    plain = {"num_layers": 8, "num_kv_heads": 4, "head_dim": 64}
    windowed = dict(plain, sliding_window=10**6)
    _same(
        kv_cache_bytes(windowed, 300),
        kv_cache_bytes(plain, 300),
        "a window larger than the context must not pad or change the cache",
    )


def test_window_caps_the_cache_once_context_exceeds_it() -> None:
    cfg = {"num_layers": 4, "num_kv_heads": 2, "head_dim": 16, "sliding_window": 128}
    capped = kv_cache_bytes(cfg, 129)
    at_window = kv_cache_bytes(cfg, 128)
    _same(
        capped,
        at_window,
        "past the window the cache must stop growing: store min(context, window) tokens",
    )


def test_none_window_means_full_attention() -> None:
    cfg = {"num_layers": 2, "num_kv_heads": 2, "head_dim": 8, "sliding_window": None}
    plain = {"num_layers": 2, "num_kv_heads": 2, "head_dim": 8}
    _same(
        kv_cache_bytes(cfg, 500),
        kv_cache_bytes(plain, 500),
        "sliding_window None must mean full attention, not a zero-length window",
    )


def test_zero_context_is_zero_bytes_for_every_dtype() -> None:
    cfg = {"num_layers": 80, "num_kv_heads": 8, "head_dim": 128}
    for dtype in BYTES:
        _same(kv_cache_bytes(cfg, 0, 3, dtype), 0, "an empty context must cost 0 bytes")


def test_batch_scales_the_cache_linearly() -> None:
    cfg = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
    one = kv_cache_bytes(cfg, 2048, 1)
    _same(
        kv_cache_bytes(cfg, 2048, 7),
        7 * one,
        "batch must multiply the whole cache, not one layer",
    )


@pytest.mark.parametrize("dtype", sorted(BYTES), ids=sorted(BYTES))
def test_each_dtype_uses_its_documented_element_width(dtype: str) -> None:
    cfg = {"num_layers": 3, "num_kv_heads": 5, "head_dim": 7}
    _same(
        kv_cache_bytes(cfg, 11, 2, dtype),
        2 * 3 * 5 * 7 * 11 * 2 * BYTES[dtype],
        "the bytes per element for this dtype are wrong: fp32 is 4, fp16 and bf16 are 2, fp8 is 1",
    )


def test_extra_config_keys_are_ignored() -> None:
    cfg = {"num_layers": 2, "num_kv_heads": 2, "head_dim": 8}
    noisy = dict(cfg, hidden_size=10**9, num_attention_heads=999, vocab_size=5)
    _same(
        kv_cache_bytes(noisy, 9),
        kv_cache_bytes(cfg, 9),
        "the result must depend only on the cache keys",
    )


def test_large_context_stays_exact_beyond_float_precision() -> None:
    cfg = {"num_layers": 80, "num_kv_heads": 64, "head_dim": 256}
    seq_len = 2**22
    batch = 2**12
    expected = 2 * 80 * 64 * 256 * seq_len * batch * 4
    assert expected > 2**53, "test setup: the expected value must exceed 2**53"
    _same(
        kv_cache_bytes(cfg, seq_len, batch, "fp32"),
        expected,
        "the result lost precision for a very large cache: keep integer arithmetic, not float",
    )


def test_config_is_not_mutated() -> None:
    cfg = {"num_layers": 2, "num_kv_heads": 2, "head_dim": 8, "sliding_window": 4}
    before = dict(cfg)
    kv_cache_bytes(cfg, 10, 2, "fp8")
    if cfg != before:
        pytest.fail("the config dict was changed: read the keys without writing to the input")


@pytest.mark.parametrize(
    ("config", "seq_len", "batch", "dtype"),
    [
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, True, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, 4.0, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, 4, True, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, 4, 1.5, "fp16"),
        ({"num_layers": 1.0, "num_kv_heads": 1, "head_dim": 1}, 4, 1, "fp16"),
        ({"num_layers": "2", "num_kv_heads": 1, "head_dim": 1}, 4, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1, "sliding_window": 2.0}, 4, 1, "fp16"),
    ],
    ids=[
        "bool-seq-len",
        "float-seq-len",
        "bool-batch",
        "float-batch",
        "float-layers",
        "string-layers",
        "float-window",
    ],
)
def test_non_int_values_raise_type_error(
    config: dict[str, Any], seq_len: Any, batch: Any, dtype: str
) -> None:
    _expect_error(
        TypeError,
        lambda: kv_cache_bytes(config, seq_len, batch, dtype),
        "a value that is not a Python int must raise TypeError",
    )


def test_config_that_is_not_a_dict_raises_type_error() -> None:
    _expect_error(
        TypeError,
        lambda: kv_cache_bytes(["num_layers"], 4),  # type: ignore[arg-type]  # invalid config
        "a config that is not a dict must raise TypeError",
    )


@pytest.mark.parametrize(
    ("config", "seq_len", "batch", "dtype"),
    [
        ({"num_kv_heads": 1, "head_dim": 1}, 4, 1, "fp16"),
        ({"num_layers": 1, "head_dim": 1}, 4, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1}, 4, 1, "fp16"),
        ({"num_layers": 0, "num_kv_heads": 1, "head_dim": 1}, 4, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": -1, "head_dim": 1}, 4, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1, "sliding_window": 0}, 4, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, -1, 1, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, 4, 0, "fp16"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, 4, 1, "int4"),
        ({"num_layers": 1, "num_kv_heads": 1, "head_dim": 1}, 4, 1, "FP16"),
    ],
    ids=[
        "missing-layers",
        "missing-kv-heads",
        "missing-head-dim",
        "zero-layers",
        "negative-kv-heads",
        "zero-window",
        "negative-context",
        "zero-batch",
        "unknown-dtype-int4",
        "upper-case-dtype",
    ],
)
def test_out_of_range_or_unknown_values_raise_value_error(
    config: dict[str, Any], seq_len: int, batch: int, dtype: str
) -> None:
    _expect_error(
        ValueError,
        lambda: kv_cache_bytes(config, seq_len, batch, dtype),
        "a missing key, an out-of-range value or an unknown dtype must raise ValueError",
    )


def test_window_of_one_token_caps_the_cache_at_one_token() -> None:
    # The smallest allowed window is 1: one stored token per layer, so
    # 2 * 1 * 1 * 1 * 1 * 1 * 2 = 4 bytes at fp16, however long the context is.
    cfg = {"num_layers": 1, "num_kv_heads": 1, "head_dim": 1, "sliding_window": 1}
    _same(
        kv_cache_bytes(cfg, 5),
        4,
        "a window of 1 is valid and must keep exactly one token per layer",
    )
