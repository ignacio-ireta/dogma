# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
from __future__ import annotations

from submission import finetune_memory_bytes


def test_bf16_base_with_a_thousand_adapter_numbers_totals_2012000_bytes():
    got = finetune_memory_bytes(
        base_params=1_000_000, base_bits=16, lora_params=1_000, adapter_bits=16
    )
    assert got == {
        "base_weights": 2_000_000,
        "adapter_weights": 2_000,
        "adapter_gradients": 2_000,
        "adam_states": 8_000,
        "total": 2_012_000,
    }, "terms must be count x bits / 8 (rounded up), and Adam costs 8 bytes per adapter number"


def test_four_bit_base_rounds_a_fractional_byte_up():
    got = finetune_memory_bytes(base_params=3, base_bits=4, lora_params=0, adapter_bits=16)
    assert got["base_weights"] == 2, "3 numbers at 4 bits is 1.5 bytes, which must round up to 2"
    assert got["total"] == 2, "with no adapter the total is the base term alone"


def test_zero_adapter_has_no_adapter_or_optimizer_bytes():
    got = finetune_memory_bytes(base_params=10, base_bits=8, lora_params=0, adapter_bits=32)
    assert got["adapter_weights"] == got["adapter_gradients"] == got["adam_states"] == 0, (
        "with no trainable numbers the adapter, gradient and Adam terms must all be zero"
    )
