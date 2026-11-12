# Hidden tests (CONTENT_SPEC §6.4). Failure messages name the kind of mistake and never show hidden
# configs or expected values. The oracle is a torch-free shape enumeration: it lists every tensor
# that exists and multiplies its shape out. It shares no formula with the learner's solution.
from __future__ import annotations

import copy
import functools
import math
import random

import pytest
from submission import count_parameters

from dogma_data import load_text

COMPONENTS = (
    "token_embedding", "position_embedding", "attention", "mlp", "norms", "final_norm", "lm_head",
)  # fmt: skip


def _shapes(cfg):
    """Enumerate (component, shape) for every tensor. One entry per tensor, all layers listed."""
    v, d, f = cfg["vocab_size"], cfg["d_model"], cfg["d_ff"]
    norm = [(d,), (d,)] if cfg["norm"] == "layernorm" else [(d,)]
    out = [("token_embedding", (v, d))]
    if cfg["learned_positions"]:
        out.append(("position_embedding", (cfg["context_length"], d)))
    for _ in range(cfg["n_layers"]):
        out += [("attention", (d, d))] * 4
        if cfg["bias"]:
            out += [("attention", (d,))] * 4
        if cfg["mlp"] == "gelu":
            out += [("mlp", (d, f)), ("mlp", (f, d))]
            if cfg["bias"]:
                out += [("mlp", (f,)), ("mlp", (d,))]
        else:  # swiglu: gate (D, F), up (D, F), down (F, D)
            out += [("mlp", (d, f)), ("mlp", (d, f)), ("mlp", (f, d))]
            if cfg["bias"]:
                out += [("mlp", (f,)), ("mlp", (f,)), ("mlp", (d,))]
        out += [("norms", s) for s in norm] * 2
    out += [("final_norm", s) for s in norm]
    if not cfg["tie_embeddings"]:
        out.append(("lm_head", (d, v)))
    return out


def _reference(cfg):
    """Per-component counts from the enumeration, with a total that is the sum of the parts."""
    counts = dict.fromkeys(COMPONENTS, 0)
    for name, shape in _shapes(cfg):
        counts[name] += math.prod(shape)
    counts["total"] = sum(counts[name] for name in COMPONENTS)
    return counts


def _base(**changes):
    cfg = {
        "vocab_size": 10, "d_model": 4, "n_layers": 2, "n_heads": 2, "d_ff": 8,
        "context_length": 6, "learned_positions": True, "bias": False, "mlp": "gelu",
        "norm": "layernorm", "tie_embeddings": True,
    }  # fmt: skip
    cfg.update(changes)
    return cfg


def _call(cfg):
    try:
        return count_parameters(cfg)
    except ValueError:
        pytest.fail("valid config was rejected: check the contract for this combination")
    except Exception as exc:
        pytest.fail(f"valid config crashed ({type(exc).__name__}): check the key names")


def _assert_counts(cfg):
    got = _call(cfg)
    if not isinstance(got, dict):
        pytest.fail("return a dict of component counts")
    if list(got) != [*COMPONENTS, "total"]:
        pytest.fail("return the eight keys in the documented order")
    if any(type(value) is not int for value in got.values()):
        pytest.fail("every count must be a Python int")
    ref = _reference(cfg)
    for name in COMPONENTS:
        if got[name] != ref[name]:
            pytest.fail(_MISTAKE_BY_COMPONENT[name])
    if got["total"] != ref["total"]:
        pytest.fail("total must be the exact sum of the other seven entries")


_MISTAKE_BY_COMPONENT = {
    "token_embedding": "token embedding count differs: the table is (V, D)",
    "position_embedding": "position table count differs: it exists only when learned, as (C, D)",
    "attention": "attention count differs: four full (D, D) projections per block, plus (D,) "
    "biases when bias is on, all multiplied by the number of layers",
    "mlp": "MLP count differs: a gelu MLP has two matrices and a swiglu MLP has three",
    "norms": "per-block norm count differs: two norms per block, each (2D) or (D) by norm type",
    "final_norm": "final norm count differs: one norm after the last block",
    "lm_head": (
        "LM head count differs: a tied head adds nothing, an untied one is (D, V) with no bias"
    ),
}


@pytest.mark.parametrize(
    "cfg",
    [
        _base(),
        _base(tie_embeddings=False),
        _base(bias=True),
        _base(bias=True, tie_embeddings=False),
        _base(learned_positions=False),
        _base(mlp="swiglu", bias=False, norm="rmsnorm", tie_embeddings=False),
        _base(mlp="swiglu", bias=True, norm="rmsnorm"),
        _base(n_layers=0),
        _base(n_layers=0, tie_embeddings=False, bias=True),
        _base(vocab_size=1, d_model=1, n_heads=1, d_ff=1, context_length=1),
    ],
    ids=[
        "tied-gelu-layernorm", "untied-head", "biased", "biased-untied", "rotary-no-table",
        "swiglu-rmsnorm-untied", "swiglu-biased-rmsnorm", "zero-layers", "zero-layers-biased",
        "all-ones",
    ],
)  # fmt: skip
def test_component_counts_match_the_shape_enumeration(cfg):
    _assert_counts(cfg)


@functools.cache
def _passages() -> tuple[tuple[int, int], ...]:
    """(vocabulary size, context length) of real passages.

    A passage is three consecutive lines of Frankenstein. Its vocabulary size is the number
    of distinct words in it and its context length is the number of words of its longest
    line: the two sizes a tiny model for that passage would be built with.
    """
    lines = [line.lower().split() for line in load_text("gutenberg_frankenstein").splitlines()]
    lines = [line for line in lines if line]
    sizes = []
    for start in range(len(lines) - 2):
        chunk = lines[start : start + 3]
        sizes.append((len({word for line in chunk for word in line}), max(map(len, chunk))))
    return tuple(sizes)


def test_seeded_random_configs_match_the_shape_enumeration():
    rng = random.Random(2026)  # noqa: S311 - seeded test data, not security
    for _ in range(40):
        heads = rng.randint(1, 4)
        d = heads * rng.randint(1, 5)
        vocab_size, context_length = rng.choice(_passages())
        cfg = _base(
            vocab_size=vocab_size,
            d_model=d,
            n_layers=rng.randint(0, 3),
            n_heads=heads,
            d_ff=rng.randint(1, 20),
            context_length=context_length,
            learned_positions=rng.random() < 0.5,
            bias=rng.random() < 0.5,
            mlp=rng.choice(["gelu", "swiglu"]),
            norm=rng.choice(["layernorm", "rmsnorm"]),
            tie_embeddings=rng.random() < 0.5,
        )
        _assert_counts(cfg)


def test_gated_mlp_counts_three_matrices_not_two():
    cfg = _base(mlp="swiglu", bias=False, n_layers=1, d_ff=8, learned_positions=False)
    got = _call(cfg)
    if got["mlp"] != 3 * 4 * 8:
        pytest.fail("gated MLP count differs: count the gate, the up and the down matrices")


def test_tied_head_adds_no_parameters():
    tied = _call(_base(tie_embeddings=True))
    untied = _call(_base(tie_embeddings=False))
    if tied["lm_head"] != 0 or untied["lm_head"] != 10 * 4:
        pytest.fail("tied head must count zero, and an untied head must count V * D")
    if untied["total"] - tied["total"] != 10 * 4:
        pytest.fail("untying must add exactly V * D parameters and nothing else")


def test_large_config_is_counted_exactly_with_python_integers():
    cfg = _base(
        vocab_size=10**9, d_model=2**13, n_layers=10**4, n_heads=8, d_ff=2**15,
        context_length=10**5, bias=True, mlp="swiglu", norm="rmsnorm", tie_embeddings=False,
    )  # fmt: skip
    got = _call(cfg)
    ref = _reference_closed_form(cfg)
    if got["total"] != ref:
        pytest.fail("large config total is not exact: keep integer arithmetic, no float division")
    if type(got["total"]) is not int:
        pytest.fail("the total must be a Python int")


def _reference_closed_form(cfg):
    d, f, n, v = cfg["d_model"], cfg["d_ff"], cfg["n_layers"], cfg["vocab_size"]
    attn = 4 * d * d + 4 * d
    mlp = 3 * d * f + 2 * f + d
    norms = 2 * d
    total = v * d + cfg["context_length"] * d * cfg["learned_positions"]
    total += n * (attn + mlp + norms) + d + v * d  # norms = two rmsnorms of (D,)
    return total


def test_inputs_are_not_mutated():
    cfg = _base()
    before = copy.deepcopy(cfg)
    _call(cfg)
    if cfg != before:
        pytest.fail("the config dict was modified: never write into the caller's config")


def _without(key):
    def build():
        cfg = _base()
        del cfg[key]
        return cfg

    return build


def _with(**changes):
    def build():
        return _base(**changes)

    return build


def _extra_key():
    cfg = _base()
    cfg["dropout"] = 0.1
    return cfg


INVALID_CASES = {
    "missing-key": _without("bias"),
    "extra-key": _extra_key,
    "float-layers": _with(n_layers=2.0),
    "bool-width": _with(
        vocab_size=True
    ),  # True == 1 is a valid size: only the type rule rejects it
    "bool-layers": _with(n_layers=False),
    "zero-vocabulary": _with(vocab_size=0),
    "heads-do-not-divide": _with(n_heads=3),
    "unknown-mlp": _with(mlp="relu"),
    "unknown-norm": _with(norm="batchnorm"),
    "int-flag": _with(bias=0),
    "not-a-dict": lambda: [_base()],
}


@pytest.mark.parametrize("case", list(INVALID_CASES), ids=list(INVALID_CASES))
def test_invalid_configs_raise_value_error(case):
    arg = INVALID_CASES[case]()
    try:
        count_parameters(arg)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"invalid config ({case}) raised {type(exc).__name__} instead of ValueError")
    pytest.fail(f"invalid config ({case}) was accepted instead of raising ValueError")


def _must_reject(cfg, message):
    try:
        count_parameters(cfg)
    except ValueError:
        return
    except Exception as exc:
        pytest.fail(f"{message} (raised {type(exc).__name__} instead of ValueError)")
    pytest.fail(message)


def test_context_length_below_one_is_rejected_with_a_position_table():
    _must_reject(
        _base(context_length=0, learned_positions=True),
        "context_length must be at least 1, so a zero-length position table is invalid",
    )


def test_context_length_below_one_is_rejected_without_a_position_table():
    _must_reject(
        _base(context_length=0, learned_positions=False),
        "context_length must be at least 1 even when no position table is counted",
    )


def test_zero_heads_is_rejected_as_a_value_error():
    _must_reject(_base(n_heads=0), "n_heads must be at least 1 and cannot be zero")


def test_negative_layer_count_is_rejected():
    _must_reject(_base(n_layers=-1), "n_layers must be at least 0, so a negative count is invalid")
