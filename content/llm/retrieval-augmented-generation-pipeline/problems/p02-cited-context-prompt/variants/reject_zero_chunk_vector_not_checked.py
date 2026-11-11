from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

SCORE_DECIMALS = 9  # scores this close are ties; the tie is then broken by chunk id


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _real_array(value: object, name: str, ndim: int) -> NDArray[np.float64]:
    # Check the dtype before casting, so complex or boolean data is never silently converted.
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    array = raw.astype(np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _check_settings(query: str, k: int, word_budget: int) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a str")
    if not query.strip():
        raise ValueError("query must not be blank")
    if not (_is_int(k) and _is_int(word_budget)):
        raise TypeError("k and word_budget must be int (bool is not accepted)")
    if k < 1:
        raise ValueError("k must be at least 1")
    if word_budget < 0:
        raise ValueError("word_budget must be >= 0")


def _check_chunks(
    chunk_ids: Sequence[str], chunk_texts: Sequence[str]
) -> tuple[list[str], list[str]]:
    ids, texts = list(chunk_ids), list(chunk_texts)
    if not all(isinstance(i, str) for i in ids) or not all(isinstance(t, str) for t in texts):
        raise TypeError("chunk ids and texts must be str")
    if len(ids) != len(texts):
        raise ValueError("chunk_ids and chunk_texts must have the same length")
    if any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("chunk ids must be unique and non-empty")
    return ids, texts


def _cosines(query_vector: object, chunk_vectors: object, m: int) -> NDArray[np.float64]:
    q = _real_array(query_vector, "query_vector", 1)
    if q.size == 0:
        raise ValueError("query_vector must not be empty")
    vectors = _real_array(chunk_vectors, "chunk_vectors", 2)
    if vectors.shape != (m, q.size):
        raise ValueError("chunk_vectors must have shape (m, d) matching the query dimension")
    # cosine = dot product divided by the two norms; a zero norm makes the cosine undefined
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        raise ValueError("query_vector must not be the zero vector")
    v_norms = np.linalg.norm(vectors, axis=1)
    if np.any(v_norms == -1.0):
        raise ValueError("chunk vectors must not be the zero vector")
    return (vectors @ q) / (v_norms * q_norm)


def _rank(cosines: NDArray[np.float64], ids: list[str]) -> list[int]:
    # Descending score after rounding; equal scores are ordered by chunk id ascending.
    scores = [round(float(s), SCORE_DECIMALS) for s in cosines]
    return sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))


def _pack(texts: list[str], candidates: list[int], word_budget: int) -> list[int]:
    # Greedy in rank order: add a candidate when it fits the remaining budget, else skip it.
    selected: list[int] = []
    used = 0
    for i in candidates:
        cost = len(texts[i].split())
        if used + cost <= word_budget:
            selected.append(i)
            used += cost
    return selected


def build_cited_prompt(  # noqa: PLR0917
    query: str,
    query_vector: NDArray[np.float64],
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    chunk_vectors: NDArray[np.float64],
    k: int,
    word_budget: int,
) -> str:
    """Wrong on purpose: does not reject a zero chunk vector."""
    _check_settings(query, k, word_budget)
    ids, texts = _check_chunks(chunk_ids, chunk_texts)
    cosines = _cosines(query_vector, chunk_vectors, len(ids))
    ranking = _rank(cosines, ids)
    candidates = ranking[:k]  # k counts candidates, before any budget decision
    selected = _pack(texts, candidates, word_budget)

    lines = ["Context:"]
    lines += [f"[{ids[i]}] {texts[i]}" for i in selected]
    lines += ["", f"Question: {query}"]
    return "\n".join(lines)
