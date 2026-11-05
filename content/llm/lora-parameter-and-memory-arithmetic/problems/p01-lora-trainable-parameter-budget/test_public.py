# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
from __future__ import annotations

import math

import pytest
from submission import lora_budget


def test_two_attention_targets_across_two_layers_give_128_trainable_parameters():
    trainable, fraction = lora_budget(
        n_layers=2, d_model=8, d_ff=None, rank=2, targets=("q", "v"), base_params=1000
    )
    assert trainable == 128, "per matrix r * (d_in + d_out) = 32; two matrices; two layers"
    assert math.isclose(fraction, 0.128, rel_tol=1e-12), "fraction must be trainable / base_params"


def test_all_seven_targets_match_the_hand_computed_budget():
    trainable, fraction = lora_budget(
        n_layers=3,
        d_model=8,
        d_ff=32,
        rank=1,
        targets=("q", "k", "v", "o", "gate", "up", "down"),
        base_params=10_000,
    )
    assert trainable == 552, "attention gives 64 per layer and each MLP matrix gives 40: check d_ff"
    assert math.isclose(fraction, 0.0552, rel_tol=1e-12), "fraction must use the same base_params"


def test_mlp_target_without_d_ff_raises_value_error():
    with pytest.raises(ValueError):
        lora_budget(n_layers=1, d_model=8, d_ff=None, rank=2, targets=("up",), base_params=100)
