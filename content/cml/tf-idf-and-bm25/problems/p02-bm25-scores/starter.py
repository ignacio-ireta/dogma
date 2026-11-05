from __future__ import annotations

from collections.abc import Sequence


def bm25_scores(
    query_tokens: Sequence[str],
    docs_tokens: Sequence[Sequence[str]],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Okapi BM25 score of every document for one query, in input order.

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
    raise NotImplementedError
