import math
import re

import numpy as np

TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tfidf_matrix(docs, *, vocabulary=None):
    if len(docs) == 0:
        raise ValueError("docs is empty")
    per_doc = []
    for doc in docs:
        tokens = TOKEN.findall(doc.lower())
        per_doc.append({t: tokens.count(t) for t in set(tokens)})
    if vocabulary is None:
        seen = set()
        for d in per_doc:
            seen.update(d)
        terms = sorted(seen)
    else:
        terms = list(vocabulary)
        if len(set(terms)) != len(terms):
            raise ValueError("repeated vocabulary term")
    if not terms:
        raise ValueError("no terms")
    n = len(docs)
    out = np.zeros((n, len(terms)), dtype=np.float64)
    for j, term in enumerate(terms):
        df = sum(1 for d in per_doc if term in d)
        idf = math.log((1 + n) / (1 + df)) + 1
        for i, d in enumerate(per_doc):
            if term in d:
                out[i, j] = d[term] * idf
    for i in range(n):
        length = math.sqrt(sum(x * x for x in out[i]))
        if length > 0:
            out[i] /= length
    return out, terms
