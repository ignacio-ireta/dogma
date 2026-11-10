from __future__ import annotations

from collections.abc import Sequence

# Input and output widths of every adaptable matrix, as functions of (d_model, d_ff).
_ATTENTION = frozenset({"q", "k", "v", "o"})
_MLP_IN = frozenset({"gate", "up"})  # d_model -> d_ff
_MLP_OUT = frozenset({"down"})  # d_ff -> d_model
_KNOWN = _ATTENTION | _MLP_IN | _MLP_OUT


def _positive_int(value: object, name: str) -> int:
    # bool is a subclass of int; a bool would silently count as 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def lora_budget(
    *,
    n_layers: int,
    d_model: int,
    d_ff: int | None,
    rank: int,
    targets: Sequence[str],
    base_params: int,
) -> tuple[int, float]:
    """Wrong on purpose: checks d_ff even when no MLP target is present."""
    n_layers = _positive_int(n_layers, "n_layers")
    d_model = _positive_int(d_model, "d_model")
    rank = _positive_int(rank, "rank")
    base_params = _positive_int(base_params, "base_params")
    if not isinstance(targets, (list, tuple)) or len(targets) == 0:
        raise ValueError("targets must be a non-empty list or tuple")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must not repeat a name")
    unknown = [t for t in targets if t not in _KNOWN]
    if unknown:
        raise ValueError("targets must come from q, k, v, o, gate, up, down")

    needs_mlp = any(t in _MLP_IN or t in _MLP_OUT for t in targets)
    ff_width = 0  # only read when an MLP target is present, and then it is validated below
    if needs_mlp or d_ff is not None:
        if d_ff is None:
            raise ValueError("an MLP target needs d_ff")
        ff_width = _positive_int(d_ff, "d_ff")

    # Step 1, one matrix at a time. A LoRA pair (A, B) holds rank * d_in + d_out * rank numbers,
    # that is rank * (d_in + d_out). The product d_in * d_out is what full fine-tuning would
    # train, and it does not appear here.
    per_layer = 0
    for t in targets:
        if t in _ATTENTION:
            d_in, d_out = d_model, d_model
        elif t in _MLP_IN:
            d_in, d_out = d_model, ff_width
        else:
            d_in, d_out = ff_width, d_model
        per_layer += rank * (d_in + d_out)

    # Step 2, every block repeats the same targets, so the per-layer sum is multiplied by n_layers.
    trainable = n_layers * per_layer

    # Step 3, the fraction is a float, computed last from exact integers.
    return trainable, trainable / base_params
