from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def packed_causal_mask(doc_ids: ArrayLike) -> NDArray[np.bool_]:
    """Return the (T, T) block-diagonal causal mask for one packed row.

    Parameters
    ----------
    doc_ids : 1-D integer array-like of length T >= 1. Each run of equal ids is one document.

    Returns
    -------
    (T, T) boolean array. ``mask[i, j]`` is True when query i may attend key j, which holds
    exactly when ``doc_ids[i] == doc_ids[j]`` and ``j <= i``.

    Raises
    ------
    ValueError
        If doc_ids is not 1-D, is empty, is not an integer dtype, or has an id that reappears
        after a different id.
    """
    raise NotImplementedError


def restart_positions(doc_ids: ArrayLike) -> NDArray[np.int64]:
    """Return the per-document position id of every token in one packed row.

    Parameters
    ----------
    doc_ids : 1-D integer array-like of length T >= 1, with the same contiguity rule as above.

    Returns
    -------
    (T,) int64 array. ``pos[i]`` is ``i`` minus the index of the first token of its document.

    Raises
    ------
    ValueError
        Under the same conditions as ``packed_causal_mask``.
    """
    raise NotImplementedError
