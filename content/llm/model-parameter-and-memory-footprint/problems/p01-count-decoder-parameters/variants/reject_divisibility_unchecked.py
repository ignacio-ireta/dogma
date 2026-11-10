from __future__ import annotations

# The exact keys of a config. A missing or extra key is an error, not a default.
CONFIG_KEYS = frozenset(
    {
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "d_ff",
        "context_length",
        "learned_positions",
        "bias",
        "mlp",
        "norm",
        "tie_embeddings",
    }
)


def _int(config: dict, key: str, minimum: int) -> int:
    """Read an int field. bool and float values are rejected, even when they look like ints."""
    value = config[key]
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an int of at least {minimum}")
    return value


def _flag(config: dict, key: str) -> bool:
    """Read a bool field, rejecting ints such as 0 and 1."""
    value = config[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a bool")
    return value


def count_parameters(config: dict) -> dict[str, int]:
    """Wrong on purpose: return the exact parameter count of a decoder-only transformer, by component.

    Shapes: token table (V, D); position table (C, D); per block, four (D, D) attention
    projections, a gelu MLP of (D, F) and (F, D) or a swiglu MLP of (D, F), (D, F) and (F, D),
    and two norms; the final norm; an LM head (D, V) when untied. Biases add (D,) or (F,) vectors.
    """  # noqa: E501
    if type(config) is not dict:
        raise ValueError("config must be a dict")
    if set(config) != CONFIG_KEYS:
        raise ValueError("config must have exactly the documented keys")

    vocab = _int(config, "vocab_size", 1)
    d_model = _int(config, "d_model", 1)
    n_layers = _int(config, "n_layers", 0)
    n_heads = _int(config, "n_heads", 1)  # noqa: F841
    d_ff = _int(config, "d_ff", 1)
    context = _int(config, "context_length", 1)
    learned = _flag(config, "learned_positions")
    bias = _flag(config, "bias")
    tied = _flag(config, "tie_embeddings")
    if config["mlp"] not in ("gelu", "swiglu"):
        raise ValueError('mlp must be "gelu" or "swiglu"')
    if config["norm"] not in ("layernorm", "rmsnorm"):
        raise ValueError('norm must be "layernorm" or "rmsnorm"')

    # One norm: a layernorm has scale and shift (2D), an rmsnorm has a scale only (D).
    norm_size = 2 * d_model if config["norm"] == "layernorm" else d_model

    # Attention: Q, K, V and O are each (D, D). With bias, each adds a (D,) vector.
    attention_block = 4 * d_model * d_model + (4 * d_model if bias else 0)

    # MLP: a gelu MLP has two matrices, a swiglu MLP has three (gate, up and down).
    # Biases: one (F,) per expansion matrix and one (D,) for the down projection.
    matrices = 2 if config["mlp"] == "gelu" else 3
    mlp_block = matrices * d_model * d_ff
    if bias:
        mlp_block += (matrices - 1) * d_ff + d_model

    # Each block has two norms, pre-attention and pre-MLP.
    norms_block = 2 * norm_size

    counts = {
        # Token table always exists. The position table exists only when learned.
        "token_embedding": vocab * d_model,
        "position_embedding": context * d_model if learned else 0,
        # Per-block counts multiplied by the number of layers.
        "attention": n_layers * attention_block,
        "mlp": n_layers * mlp_block,
        "norms": n_layers * norms_block,
        # The final norm sits after the last block. The head is absent when it is tied.
        "final_norm": norm_size,
        "lm_head": 0 if tied else vocab * d_model,
    }
    counts["total"] = sum(counts.values())
    return counts
