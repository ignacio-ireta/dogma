from __future__ import annotations

import pytest
import torch
from submission import build_adamw_groups

LR = 0.05


def _named(seed: int) -> list[tuple[str, torch.Tensor]]:
    """A small model: embedding, two linear layers with biases, a norm, and a conv kernel."""
    gen = torch.Generator().manual_seed(seed)

    def make(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=gen, dtype=torch.float64)

    return [
        ("embed.weight", make(5, 3)),
        ("layer1.weight", make(3, 3)),
        ("layer1.bias", make(3)),
        ("norm.weight", make(3)),
        ("conv.weight", make(2, 2, 3, 3)),
        ("proj.bias", make(4, 4)),  # a bias stored with two dimensions is still a bias
        ("layer2.weight", make(3, 2)),
        ("layer2.bias", make(2)),
    ]


def _expected_membership(named):
    decayed = []
    undecayed = []
    for name, param in named:
        if param.ndim < 2 or name.rsplit(".", 1)[-1] == "bias":
            undecayed.append(name)
        else:
            decayed.append(name)
    return decayed, undecayed


def _names_of(group, named):
    by_id = {id(tensor): name for name, tensor in named}
    return [by_id.get(id(t), "<copy or unknown tensor>") for t in group["params"]]


def _named_dict_check(named, groups) -> None:
    decayed_expected, undecayed_expected = _expected_membership(named)
    if len(groups) != 2:
        pytest.fail("expected a decayed group and an undecayed group", pytrace=False)
    decay_group, no_decay_group = groups
    if _names_of(decay_group, named) != decayed_expected:
        pytest.fail(
            "a bias or a 1-D norm gain received weight decay, or a matrix was left out",
            pytrace=False,
        )
    if _names_of(no_decay_group, named) != undecayed_expected:
        pytest.fail("biases and 1-D gains must be in the undecayed group", pytrace=False)


def test_membership_follows_rank_and_bias_name():
    named = _named(0)
    groups = build_adamw_groups(named, weight_decay=0.1)
    _named_dict_check(named, groups)


def test_group_weight_decay_values_are_exact():
    named = _named(1)
    decay_group, no_decay_group = build_adamw_groups(named, weight_decay=0.3)
    if decay_group["weight_decay"] != 0.3:
        pytest.fail("the decayed group must use the requested weight_decay", pytrace=False)
    if no_decay_group["weight_decay"] != 0.0:
        pytest.fail("the undecayed group must have weight_decay 0.0", pytrace=False)


def test_groups_reference_the_callers_tensors_not_copies():
    named = _named(2)
    groups = build_adamw_groups(named, weight_decay=0.1)
    originals = {id(t) for _, t in named}
    for group in groups:
        for tensor in group["params"]:
            if id(tensor) not in originals:
                pytest.fail(
                    "groups must reference the caller's tensors: copies never get updated",
                    pytrace=False,
                )


def test_parameter_order_is_preserved_within_each_group():
    named = _named(3)
    groups = build_adamw_groups(named, weight_decay=0.1)
    decay_names = _names_of(groups[0], named)
    undecay_names = _names_of(groups[1], named)
    if decay_names != [n for n, t in named if t.ndim >= 2 and n.rsplit(".", 1)[-1] != "bias"]:
        pytest.fail("decayed parameters must keep their original order", pytrace=False)
    if undecay_names != [
        n for n, t in named if not (t.ndim >= 2 and n.rsplit(".", 1)[-1] != "bias")
    ]:
        pytest.fail("undecayed parameters must keep their original order", pytrace=False)


def test_all_biases_and_gains_give_one_undecayed_group():
    named = [("a.bias", torch.zeros(2)), ("norm.weight", torch.ones(3))]
    groups = build_adamw_groups(named, weight_decay=0.2)
    if len(groups) != 1 or groups[0]["weight_decay"] != 0.0:
        pytest.fail(
            "only undecayed parameters must give one group with weight_decay 0.0", pytrace=False
        )


def test_adamw_run_with_these_groups_matches_torch_adamw_with_reference_groups():
    seed = 9
    named_a = _named(seed)
    named_b = _named(seed)  # identical values, separate tensors for the reference run
    learner_groups = build_adamw_groups(named_a, weight_decay=0.2)
    reference_groups = _reference_groups(named_b, weight_decay=0.2)
    opt_a = torch.optim.AdamW(learner_groups, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    opt_b = torch.optim.AdamW(reference_groups, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator().manual_seed(seed + 100)
    for _ in range(8):
        for (_, a), (_, b) in zip(named_a, named_b, strict=True):
            grad = torch.randn(a.shape, generator=gen, dtype=torch.float64)
            a.grad = grad.clone()
            b.grad = grad.clone()
        opt_a.step()
        opt_b.step()
    for (name, a), (_, b) in zip(named_a, named_b, strict=True):
        if not torch.allclose(a, b, rtol=1e-10, atol=1e-12):
            pytest.fail(
                f"{name}: AdamW with these groups differs from the reference grouping",
                pytrace=False,
            )


def _reference_groups(named, weight_decay):
    decayed = [t for n, t in named if t.ndim >= 2 and n.rsplit(".", 1)[-1] != "bias"]
    rest = [t for n, t in named if not (t.ndim >= 2 and n.rsplit(".", 1)[-1] != "bias")]
    return [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": rest, "weight_decay": 0.0},
    ]


def test_empty_input_raises_value_error():
    _rejects(lambda: build_adamw_groups([], weight_decay=0.1), "empty parameter list")


def test_duplicate_names_raise_value_error():
    named = [("w", torch.zeros(2, 2)), ("w", torch.zeros(2, 2))]
    _rejects(lambda: build_adamw_groups(named, weight_decay=0.1), "duplicate parameter names")


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")], ids=["negative", "nan", "inf"])
def test_invalid_weight_decay_raises_value_error(bad):
    named = _named(4)
    _rejects(lambda: build_adamw_groups(named, weight_decay=bad), "invalid weight_decay")


def test_non_tensor_value_raises_value_error():
    _rejects(
        lambda: build_adamw_groups([("w.weight", [[1.0]])], weight_decay=0.1),  # type: ignore[list-item]
        "value that is not a tensor",
    )


def test_input_list_and_tensors_are_not_mutated():
    named = _named(5)
    before = [(n, t.clone()) for n, t in named]
    build_adamw_groups(named, weight_decay=0.1)
    if [n for n, _ in named] != [n for n, _ in before]:
        pytest.fail("the input list of names was reordered or changed", pytrace=False)
    for (_, a), (_, b) in zip(named, before, strict=True):
        if not torch.equal(a, b):
            pytest.fail(
                "input tensors were changed: grouping must not write to them", pytrace=False
            )


def _rejects(call, case: str) -> None:
    try:
        call()
    except ValueError:
        return
    except Exception:
        pytest.fail(f"{case}: must raise ValueError, not another exception type", pytrace=False)
    pytest.fail(f"{case}: invalid input was accepted", pytrace=False)


def test_bias_inside_a_longer_name_does_not_count_as_a_bias():
    # Only the final dotted component decides. "bias" as a substring or a prefix does not.
    named = [
        ("attn.bias_scale.weight", torch.zeros(2, 2)),
        ("head.lm_bias", torch.zeros(2, 2)),
        ("bias_gate.weight", torch.zeros(3, 3)),
        ("norm.bias", torch.zeros(3)),
    ]
    groups = build_adamw_groups(named, weight_decay=0.1)
    _named_dict_check(named, groups)


def test_zero_weight_decay_is_accepted():
    named = [("linear.weight", torch.zeros(2, 2)), ("linear.bias", torch.zeros(2))]
    try:
        groups = build_adamw_groups(named, weight_decay=0.0)
    except ValueError:
        pytest.fail("weight_decay of exactly zero is valid and must not be rejected", pytrace=False)
    if groups[0]["weight_decay"] != 0.0:
        pytest.fail("with zero weight decay the decayed group must carry 0.0", pytrace=False)


def test_names_without_a_dot_follow_the_same_decay_rule():
    w = torch.zeros(2, 2)
    b = torch.zeros(2)
    s = torch.zeros(2, 2)
    named = [("weight", w), ("bias", b), ("bias_matrix", s)]
    groups = build_adamw_groups(named, weight_decay=0.1)
    decay = groups[0]["params"]
    no_decay = groups[-1]["params"]
    if len(decay) != 2 or decay[0] is not w or decay[1] is not s:
        pytest.fail(
            "a name with no dot must still be classified by its rank and its whole name",
            pytrace=False,
        )
    if len(no_decay) != 1 or no_decay[0] is not b:
        pytest.fail(
            "a one-dimensional tensor named bias without a dot must be undecayed",
            pytrace=False,
        )


def test_parameter_objects_are_accepted_and_kept_by_identity():
    # nn.Parameter is a torch.Tensor, so the values from model.named_parameters() are valid
    weight = torch.nn.Parameter(torch.zeros(2, 3))
    try:
        groups = build_adamw_groups([("linear.weight", weight)], weight_decay=0.1)
    except ValueError:
        raise AssertionError("an nn.Parameter is a torch.Tensor and must be accepted") from None
    assert groups[0]["params"][0] is weight, (
        "the decayed group must hold the parameter object itself, not a copy"
    )


def test_duplicate_names_on_one_dimensional_tensors_are_rejected():
    # names must be unique whatever the rank of the tensor they belong to
    gain = torch.ones(3)
    bias = torch.zeros(3)
    try:
        build_adamw_groups([("norm.weight", gain), ("norm.weight", bias)], weight_decay=0.1)
    except ValueError:
        return
    raise AssertionError("duplicate names on one-dimensional tensors must raise ValueError")
