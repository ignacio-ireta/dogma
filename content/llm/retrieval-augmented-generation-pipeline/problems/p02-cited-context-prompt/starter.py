from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def build_cited_prompt(  # noqa: PLR0917
    query: str,
    query_vector: NDArray[np.float64],
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    chunk_vectors: NDArray[np.float64],
    k: int,
    word_budget: int,
) -> str:
    """Rank chunks by cosine similarity, keep the top k, and pack a cited prompt.

    Scores are cosines rounded to 9 decimals; ties sort by chunk id ascending. The first
    min(k, m) ranked chunks are candidates. Walk them in rank order and add a chunk when its
    len(text.split()) words still fit word_budget, skipping (not stopping at) chunks that do not.
    Return "\\n".join of: "Context:", one "[id] text" line per selected chunk in rank order, an
    empty line, and "Question: <query>". Raise TypeError for wrong types (bool is not an int),
    ValueError for blank query, bad arrays or shapes, zero-norm vectors, duplicate or empty ids,
    k < 1 or word_budget < 0. Never mutate inputs.
    """
    raise NotImplementedError
