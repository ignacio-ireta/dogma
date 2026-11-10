from __future__ import annotations

_KEYS = {
    "vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "context_length",
    "learned_positions", "bias", "mlp", "norm", "tie_embeddings",
}  # fmt: skip
_INT_MINIMUMS = {
    "vocab_size": 1,
    "d_model": 1,
    "n_layers": 0,
    "n_heads": 1,
    "d_ff": 1,
    "context_length": 1,
}
_BOOL_KEYS = ("learned_positions", "bias", "tie_embeddings")


def count_parameters(config: dict) -> dict[str, int]:  # noqa: PLR0912, PLR0915
    """Return the exact parameter count of a decoder-only transformer, by component.

    config has exactly these keys: vocab_size, d_model, n_layers, n_heads, d_ff, context_length,
    learned_positions (bool), bias (bool), mlp ("gelu" | "swiglu"), norm ("layernorm" | "rmsnorm"),
    tie_embeddings (bool). Keys of the result, in order: token_embedding, position_embedding,
    attention, mlp, norms, final_norm, lm_head, total. Each component counts all layers.
    Raise ValueError for invalid configs. Use no numeric library and do not mutate config.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    keys = set(config.keys())
    if keys != _KEYS:
        missing = sorted(_KEYS - keys)
        extra = sorted(keys - _KEYS, key=str)
        raise ValueError(f"config keys wrong: missing {missing}, unexpected {extra}")

    for name, minimum in _INT_MINIMUMS.items():
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an int")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    for name in _BOOL_KEYS:
        if not isinstance(config[name], bool):
            raise ValueError(f"{name} must be a bool")
    if not isinstance(config["mlp"], str):
        raise ValueError('mlp must be "gelu" or "swiglu"')
    if config["mlp"] not in ("gelu", "swiglu"):
        raise ValueError('mlp must be "gelu" or "swiglu"')
    if not isinstance(config["norm"], str):
        raise ValueError('norm must be "layernorm" or "rmsnorm"')
    if config["norm"] not in ("layernorm", "rmsnorm"):
        raise ValueError('norm must be "layernorm" or "rmsnorm"')

    V = config["vocab_size"]
    D = config["d_model"]
    N = config["n_layers"]
    H = config["n_heads"]
    F = config["d_ff"]
    C = config["context_length"]
    learned = config["learned_positions"]
    bias = config["bias"]
    gated = config["mlp"] == "swiglu"
    layernorm = config["norm"] == "layernorm"
    tied = config["tie_embeddings"]

    if D % H != 0:
        raise ValueError("n_heads must divide d_model")

    token_embedding = V * D
    position_embedding = C * D if learned else 0

    # Attention: four full-width projections per block, each with an optional bias.
    attention_per_block = 4 * D * D + (4 * D if bias else 0)

    # MLP: gelu has (D, F) and (F, D); swiglu adds a (D, F) gate. Biases: F per expansion, D for down.  # noqa: E501
    if gated:
        mlp_per_block = 3 * D * F + ((2 * F + D) if bias else 0)
    else:
        mlp_per_block = 2 * D * F + ((F + D) if bias else 0)

    # Norms: two per block, each (D,) for rmsnorm or (D,) scale plus (D,) shift for layernorm.
    norm_params = 2 * D if layernorm else D
    norms_per_block = 2 * norm_params

    final_norm = norm_params
    lm_head = 0 if tied else D * V

    counts = {
        "token_embedding": token_embedding,
        "position_embedding": position_embedding,
        "attention": N * attention_per_block,
        "mlp": N * mlp_per_block,
        "norms": N * norms_per_block,
        "final_norm": final_norm,
        "lm_head": lm_head,
    }
    total = sum(counts.values())
    result: dict[str, int] = dict(counts)
    result["total"] = total
    return result
