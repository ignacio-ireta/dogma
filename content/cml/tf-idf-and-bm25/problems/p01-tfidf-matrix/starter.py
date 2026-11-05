from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def tfidf_matrix(
    docs: Sequence[str],
    *,
    vocabulary: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Smoothed TF-IDF matrix with l2-normalised rows, as scikit-learn builds it by default.

    Tokens: lowercase, then the matches of (?u)\\b\\w\\w+\\b (two or more word characters).
    tf is the raw count. idf(t) = ln((1 + n) / (1 + df(t))) + 1, with df over all documents.
    Each row is weighted by tf * idf, then divided by its l2 norm (a zero row stays zero).

    vocabulary: None for every token in docs, sorted ascending; otherwise the given distinct terms,
    in the given order, and other tokens are ignored.

    Returns
    -------
    (matrix, vocabulary): a float64 array of shape (n_docs, n_terms) and the list of column terms.

    Raises
    ------
    ValueError
        For an empty docs sequence, a vocabulary with a repeated term, or an empty vocabulary.
    """
    raise NotImplementedError
