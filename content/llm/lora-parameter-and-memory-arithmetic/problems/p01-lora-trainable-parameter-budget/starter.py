from __future__ import annotations

from collections.abc import Sequence


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
    raise NotImplementedError
