# Public tests: the examples from problem.md plus edge cases (CONTENT_SPEC §6.4).
# The learner's code is imported from submission.py; tests never import solution.py.
import pytest
from submission import count_parameters

TINY = {
    "vocab_size": 10, "d_model": 4, "n_layers": 1, "n_heads": 2, "d_ff": 8,
    "context_length": 6, "learned_positions": True, "bias": False, "mlp": "gelu",
    "norm": "layernorm", "tie_embeddings": True,
}  # fmt: skip

GPT2_SMALL = {
    "vocab_size": 50257, "d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072,
    "context_length": 1024, "learned_positions": True, "bias": True, "mlp": "gelu",
    "norm": "layernorm", "tie_embeddings": True,
}  # fmt: skip


def _call(config):
    """Call the learner; a crash on a valid config is reported with the likely cause."""
    try:
        return count_parameters(config)
    except ValueError:
        raise
    except Exception as exc:
        pytest.fail(
            f"valid config crashed ({type(exc).__name__}): check the key names and the "
            "per-component formulas"
        )


TINY_EXPECTED = {
    "token_embedding": 40, "position_embedding": 24, "attention": 64, "mlp": 64,
    "norms": 16, "final_norm": 8, "lm_head": 0, "total": 216,
}  # fmt: skip

TINY_MISTAKES = {
    "token_embedding": "token embedding differs: the table is (V, D)",
    "position_embedding": "position table differs: learned (C, D) only, and C is context_length",
    "attention": "attention differs: four (D, D) projections per block",
    "mlp": "MLP differs: a gelu MLP has two matrices, (D, F) and (F, D)",
    "norms": "per-block norms differ: two norms per block, each (2D) for layernorm",
    "final_norm": "final norm differs: one norm after the last block",
    "lm_head": "LM head differs: a tied head adds no parameters",
    "total": "total differs: it must be the sum of the other seven entries",
}


def test_tiny_tied_model_components_and_total():
    counts = _call(TINY)
    for name, expected in TINY_EXPECTED.items():
        assert counts.get(name) == expected, TINY_MISTAKES[name]


def test_gpt2_small_total_matches_the_published_configuration():
    counts = _call(GPT2_SMALL)
    assert counts["total"] == 124_439_808, (
        "GPT-2 small total differs: check bias vectors, the two norms per block and the tied head"
    )


def test_zero_layers_have_no_per_block_parameters():
    counts = _call({**TINY, "n_layers": 0})
    assert counts["attention"] == 0 and counts["mlp"] == 0 and counts["norms"] == 0, (
        "with zero layers the per-block components must all be zero"
    )


def test_unknown_mlp_name_raises_value_error():
    try:
        count_parameters({**TINY, "mlp": "relu"})
    except ValueError:
        return
    pytest.fail('mlp "relu" was accepted: the allowed names are "gelu" and "swiglu"')
