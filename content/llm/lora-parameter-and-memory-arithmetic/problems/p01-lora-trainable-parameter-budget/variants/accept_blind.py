from __future__ import annotations

from collections.abc import Sequence

_ATTENTION = ("q", "k", "v", "o")
_MLP = ("gate", "up", "down")
_KNOWN = _ATTENTION + _MLP


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_positive(name: str, value: object) -> None:
    if not _is_int(value):
        raise ValueError(f"{name} must be an int (not bool)")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def lora_budget(
    *,
    n_layers: int,
    d_model: int,
    d_ff: int | None,
    rank: int,
    targets: Sequence[str],
    base_params: int,
) -> tuple[int, float]:
    """Return (trainable, fraction) for a LoRA configuration on a decoder-only transformer.

    Each adapted matrix with input width d_in and output width d_out holds rank * (d_in + d_out)
    parameters. Attention targets q, k, v, o are d_model x d_model. Targets gate and up map
    d_model -> d_ff; down maps d_ff -> d_model. Every layer has the same targets.

    Parameters
    ----------
    n_layers, d_model, rank : positive Python ints (not bool).
    d_ff : positive int, required when gate, up or down is a target; otherwise may be None.
    targets : non-empty list or tuple of distinct names from q, k, v, o, gate, up, down.
    base_params : positive int, the total parameter count of the frozen base model.

    Returns
    -------
    trainable : int, exact, n_layers * sum of rank * (d_in + d_out) over the targets.
    fraction : float, trainable / base_params.

    Raises
    ------
    ValueError
        On a non-int or bool argument, an out-of-range value, an empty, duplicated or unknown
        target, or an MLP target with d_ff missing.
    """
    _check_positive("n_layers", n_layers)
    _check_positive("d_model", d_model)
    _check_positive("rank", rank)
    _check_positive("base_params", base_params)

    if not isinstance(targets, (list, tuple)):
        raise ValueError("targets must be a list or tuple")
    if len(targets) == 0:
        raise ValueError("targets must not be empty")
    for name in targets:
        if not isinstance(name, str) or name not in _KNOWN:
            raise ValueError(f"unknown target {name!r}")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must not repeat a name")

    needs_d_ff = any(name in _MLP for name in targets)
    if needs_d_ff:
        if d_ff is None:
            raise ValueError("an MLP target needs d_ff")
        _check_positive("d_ff", d_ff)

    widths = {
        "q": (d_model, d_model),
        "k": (d_model, d_model),
        "v": (d_model, d_model),
        "o": (d_model, d_model),
        "gate": (d_model, d_ff),
        "up": (d_model, d_ff),
        "down": (d_ff, d_model),
    }

    per_layer = 0
    for name in targets:
        d_in, d_out = widths[name]
        per_layer += rank * (d_in + d_out)

    trainable = n_layers * per_layer
    fraction = trainable / base_params
    return trainable, fraction
