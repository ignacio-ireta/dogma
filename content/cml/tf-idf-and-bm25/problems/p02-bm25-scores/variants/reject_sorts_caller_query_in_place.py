from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def bm25_scores(
    query_tokens: Sequence[str],
    docs_tokens: Sequence[Sequence[str]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Wrong on purpose: the caller's query list is sorted in place.

    query_tokens: tokens of the query; each occurrence counts. docs_tokens: non-empty
    collection of token sequences (documents may be empty). N = number of documents.
    avgdl = mean length over all documents. idf(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5)),
    where n(q) counts the documents that contain q. Term part:
    idf * f * (k1 + 1) / (f + k1 * (1 - b + b * |D| / avgdl)), zero when f = 0.

    Returns
    -------
    A list of N floats, one per document.

    Raises
    ------
    ValueError
        For an empty docs_tokens, a negative k1, or a b outside [0, 1].
    """
    if isinstance(query_tokens, list):
        query_tokens.sort()
    if len(docs_tokens) == 0:
        raise ValueError("at least one document is required")
    if k1 < 0:
        raise ValueError("k1 must not be negative")
    if not 0.0 <= b <= 1.0:
        raise ValueError("b must lie in [0, 1]")

    n_docs = len(docs_tokens)
    # One Counter per document: f(q, D) is a dictionary lookup, and the length is len(tokens).
    bags = [Counter(doc) for doc in docs_tokens]
    lengths = [len(doc) for doc in docs_tokens]
    avgdl = sum(lengths) / n_docs  # empty documents count, so this may be 0.0

    # n(q) is the number of documents that contain the token. Each occurrence of a query token adds
    # its own term part, so a repeated query token counts twice.
    scores = [0.0] * n_docs
    for token in query_tokens:  # every occurrence adds its own contribution
        containing = [j for j in range(n_docs) if bags[j][token] > 0]
        if not containing:
            continue  # an unseen term contributes nothing to any document
        n_q = len(containing)
        # Lucene's idf: the +1 inside the log keeps it positive even for a term in most documents.
        idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
        for j in containing:
            f = bags[j][token]
            # Saturation in f and length normalisation through b, both inside the denominator.
            norm = 1.0 - b + b * lengths[j] / avgdl
            scores[j] += idf * f * (k1 + 1.0) / (f + k1 * norm)
    return scores
