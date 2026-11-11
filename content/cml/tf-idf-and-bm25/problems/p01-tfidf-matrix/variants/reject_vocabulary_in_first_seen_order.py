from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

# The token rule, as scikit-learn's default token_pattern: lowercase text, then runs of two or more
# word characters. Single characters such as "a" or "7" never become tokens.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tfidf_matrix(
    docs: Sequence[str],
    *,
    vocabulary: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Wrong on purpose: the default vocabulary keeps first-seen order instead of sorting.

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
    if len(docs) == 0:
        raise ValueError("at least one document is required")

    # Step 1: tokens and raw counts per document. The regex runs on the lowercased text.
    counts = [Counter(_TOKEN.findall(doc.lower())) for doc in docs]

    # Step 2: the columns. A given vocabulary fixes the order; otherwise sort every token seen.
    if vocabulary is None:
        terms = list(dict.fromkeys(term for doc in counts for term in doc))
    else:
        terms = list(vocabulary)
        if len(set(terms)) != len(terms):
            raise ValueError("vocabulary must not repeat a term")
    if not terms:
        raise ValueError("empty vocabulary: no document contains a token of two or more characters")
    column = {term: j for j, term in enumerate(terms)}

    # Step 3: document frequency over the whole corpus, then the smoothed idf of each column.
    n_docs = len(docs)
    df = Counter(term for doc in counts for term in doc)
    idf = np.array([math.log((1 + n_docs) / (1 + df[term])) + 1.0 for term in terms])

    # Step 4: raw count times idf, placed in the column of each term that is in the vocabulary.
    matrix = np.zeros((n_docs, len(terms)), dtype=np.float64)
    for i, doc in enumerate(counts):
        for term, count in doc.items():
            j = column.get(term)
            if j is not None:
                matrix[i, j] = count * idf[j]

    # Step 5: l2-normalise each row. A row with no weight has norm 0 and is left as zeros.
    norms = np.sqrt((matrix**2).sum(axis=1, keepdims=True))
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix, terms
