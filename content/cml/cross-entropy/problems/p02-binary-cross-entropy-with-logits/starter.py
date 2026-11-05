from __future__ import annotations

from numpy.typing import ArrayLike


def binary_cross_entropy_with_logits(z: ArrayLike, y: ArrayLike) -> float:
    """Return the mean binary cross-entropy of logits z against targets y in [0, 1].

    Uses the stable per-example form max(z, 0) - z * y + log(1 + exp(-|z|)), so that no
    probability is formed and no exponent is positive.

    Parameters
    ----------
    z : 1-D array-like of finite logits, length n >= 1.
    y : 1-D array-like of finite targets in [0, 1], the same length as z.

    Returns
    -------
    float: the mean loss over the n examples, never negative.

    Raises
    ------
    ValueError
        For non-1-D inputs, a length mismatch, an empty z, non-finite logits, or targets
        outside [0, 1].
    """
    raise NotImplementedError
