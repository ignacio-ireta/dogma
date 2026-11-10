from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated_ids(doc_ids: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    arr = np.asarray(doc_ids)
    if arr.ndim != 1:
        raise ValueError("doc_ids must be 1-D")
    if arr.size == 0:
        raise ValueError("doc_ids must not be empty")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("doc_ids must be an integer dtype")
    arr = arr.astype(np.int64)
    starts = np.ones(arr.shape[0], dtype=bool)
    starts[1:] = arr[1:] != arr[:-1]
    start_values = arr[starts]
    if np.unique(start_values).size != start_values.size:
        raise ValueError("doc_ids must form contiguous runs")
    return arr, starts


def packed_causal_mask(doc_ids: ArrayLike) -> NDArray[np.bool_]:
    ids, _ = _validated_ids(doc_ids)
    idx = np.arange(ids.shape[0])
    causal = idx[:, None] >= idx[None, :]
    same_doc = ids[:, None] == ids[None, :]
    return np.asarray(causal & same_doc, dtype=np.bool_)


def restart_positions(doc_ids: ArrayLike) -> NDArray[np.int64]:
    _, starts = _validated_ids(doc_ids)
    idx = np.arange(starts.shape[0], dtype=np.int64)
    first_index = np.maximum.accumulate(np.where(starts, idx, 0))
    return (idx - first_index).astype(np.int64)
