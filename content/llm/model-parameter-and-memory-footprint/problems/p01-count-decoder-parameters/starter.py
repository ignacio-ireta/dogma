from __future__ import annotations


def count_parameters(config: dict) -> dict[str, int]:
    """Return the exact parameter count of a decoder-only transformer, by component.

    config has exactly these keys: vocab_size, d_model, n_layers, n_heads, d_ff, context_length,
    learned_positions (bool), bias (bool), mlp ("gelu" | "swiglu"), norm ("layernorm" | "rmsnorm"),
    tie_embeddings (bool). Keys of the result, in order: token_embedding, position_embedding,
    attention, mlp, norms, final_norm, lm_head, total. Each component counts all layers.
    Raise ValueError for invalid configs. Use no numeric library and do not mutate config.
    """
    raise NotImplementedError
