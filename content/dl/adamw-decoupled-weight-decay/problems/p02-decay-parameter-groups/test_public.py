from __future__ import annotations

import pytest
import torch
from submission import build_adamw_groups


def test_matrix_decays_while_bias_and_norm_gain_do_not():
    weight = torch.zeros(2, 3)
    bias = torch.zeros(2)
    gain = torch.ones(2)
    groups = build_adamw_groups(
        [("linear.weight", weight), ("linear.bias", bias), ("norm.weight", gain)],
        weight_decay=0.1,
    )
    if len(groups) != 2:
        pytest.fail("expected one decayed group and one undecayed group", pytrace=False)
    decay, no_decay = groups
    if decay["weight_decay"] != 0.1 or no_decay["weight_decay"] != 0.0:
        pytest.fail(
            "decayed group must carry weight_decay and undecayed group must carry 0.0",
            pytrace=False,
        )
    if decay["params"][0] is not weight:
        pytest.fail("the decayed group must hold the caller's weight tensor itself", pytrace=False)
    if [id(t) for t in no_decay["params"]] != [id(bias), id(gain)]:
        pytest.fail(
            "biases and 1-D norm gains must be undecayed, in their original order", pytrace=False
        )


def test_only_decayed_parameters_give_a_single_group():
    weight = torch.zeros(4, 4)
    groups = build_adamw_groups([("embed.weight", weight)], weight_decay=0.05)
    if len(groups) != 1 or groups[0]["weight_decay"] != 0.05:
        pytest.fail("a single decayed parameter must give exactly one group", pytrace=False)


def test_empty_parameter_list_raises_value_error():
    with pytest.raises(ValueError):
        build_adamw_groups([], weight_decay=0.1)
