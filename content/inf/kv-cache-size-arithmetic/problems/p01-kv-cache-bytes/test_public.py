"""Public tests: the examples from problem.md, one empty-context edge case, and the dtype rule."""

from __future__ import annotations

from typing import Any

import pytest
from submission import kv_cache_bytes

MHA_7B = {"num_layers": 32, "num_kv_heads": 32, "head_dim": 128}
GQA_WINDOWED = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "sliding_window": 4096}


def _int_bytes(got: Any) -> None:
    if type(got) is not int:
        pytest.fail("return a Python int of bytes, not a float; a GiB value is a float, not bytes")


def _expect_value_error(call: Any, message: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{message}: raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"{message}: no error was raised")


def test_seven_b_multi_head_fp16_is_two_gib_per_sequence() -> None:
    got = kv_cache_bytes(MHA_7B, 4096)
    _int_bytes(got)
    assert got == 2 * 2**30, (
        "wrong size for one 4096-token sequence: check the factor 2 for K and V, and that "
        "fp16 uses 2 bytes per element"
    )


def test_sliding_window_caps_stored_tokens_and_uses_kv_heads() -> None:
    got = kv_cache_bytes(GQA_WINDOWED, 32768)
    _int_bytes(got)
    assert got == 512 * 2**20, (
        "wrong size with 8 KV heads and a 4096-token window: store min(context, window) tokens, "
        "and count KV heads rather than query heads"
    )


def test_zero_context_has_no_cache_for_any_batch() -> None:
    assert kv_cache_bytes(MHA_7B, 0, batch=5) == 0, (
        "an empty context must cost 0 bytes: the context length multiplies the whole formula"
    )


def test_unknown_dtype_is_rejected_with_value_error() -> None:
    _expect_value_error(
        lambda: kv_cache_bytes(MHA_7B, 16, dtype="int4"),
        "dtype outside fp32, fp16, bf16 and fp8 must raise ValueError",
    )
