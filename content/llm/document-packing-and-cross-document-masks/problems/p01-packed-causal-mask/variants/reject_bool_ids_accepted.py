from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _checked_doc_ids(doc_ids: ArrayLike) -> NDArray[np.integer]:
    """Validate the document ids and return them. The caller's array is never written to."""
    ids = np.asarray(doc_ids)
    if ids.ndim != 1:
        raise ValueError(f"doc_ids must be 1-D, got shape {ids.shape}")
    if ids.shape[0] < 1:
        raise ValueError("doc_ids must hold at least one token")
    # bool is a separate kind in NumPy, so True/False is rejected along with floats.
    if ids.dtype.kind not in "iub":
        raise ValueError("doc_ids must be integers; bool and float are rejected")
    # A document is a run of equal ids. Its id must not come back after another document,
    # otherwise the equality rule below would let a later run attend an earlier one.
    run_starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    run_ids = ids[run_starts]
    if np.unique(run_ids).size != run_ids.size:
        raise ValueError("doc_ids must form contiguous runs; an id reappears after another")
    return ids


def _run_start_index(ids: NDArray[np.integer]) -> NDArray[np.int64]:
    """For each token, the index of the first token of its run (its document start)."""
    n = ids.shape[0]
    is_start = np.r_[True, ids[1:] != ids[:-1]]
    # Keep the index only where a run begins, else 0. Starts increase along the row, so a running
    # maximum carries each run's start forward to the tokens inside it.
    marks = np.where(is_start, np.arange(n), 0)
    return np.maximum.accumulate(marks).astype(np.int64)


def packed_causal_mask(doc_ids: ArrayLike) -> NDArray[np.bool_]:
    """Wrong on purpose: boolean ids are accepted as integers.
    Return the (T, T) block-diagonal causal mask: mask[i, j] = (doc[i] == doc[j]) and j <= i."""
    ids = _checked_doc_ids(doc_ids)
    positions = np.arange(ids.shape[0])
    # Two separate conditions, combined with AND. The same-document rule blocks cross-document
    # keys; the causal rule blocks future keys. A token always passes both with itself.
    same_document = ids[:, None] == ids[None, :]
    causal = positions[None, :] <= positions[:, None]
    return same_document & causal


def restart_positions(doc_ids: ArrayLike) -> NDArray[np.int64]:
    """Return pos[i] = i minus the index of the first token of its document."""
    ids = _checked_doc_ids(doc_ids)
    return (np.arange(ids.shape[0]) - _run_start_index(ids)).astype(np.int64)
