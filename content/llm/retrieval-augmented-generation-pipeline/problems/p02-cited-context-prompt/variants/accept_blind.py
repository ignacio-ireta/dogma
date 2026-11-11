import numpy as np


def build_cited_prompt(query, query_vector, chunk_ids, chunk_texts, chunk_vectors, k, word_budget):  # noqa: PLR0912, PLR0917
    if not isinstance(query, str):
        raise TypeError("query must be a str")
    if query.strip() == "":
        raise ValueError("blank query")
    for value in (k, word_budget):
        if type(value) is not int:
            raise TypeError("k and word_budget must be int")
    if k < 1 or word_budget < 0:
        raise ValueError("k >= 1 and word_budget >= 0")
    ids = list(chunk_ids)
    texts = list(chunk_texts)
    if not all(isinstance(i, str) for i in ids) or not all(isinstance(x, str) for x in texts):
        raise TypeError("ids and texts must be str")
    if len(ids) != len(texts) or len(set(ids)) != len(ids) or any(i == "" for i in ids):
        raise ValueError("bad ids")
    raw_q = np.asarray(query_vector)
    raw_v = np.asarray(chunk_vectors)
    if raw_q.dtype.kind not in "iuf" or raw_v.dtype.kind not in "iuf":
        raise ValueError("vectors must be real numeric")
    q = raw_q.astype(float)
    vecs = raw_v.astype(float)
    if q.ndim != 1 or vecs.shape != (len(ids), q.shape[0]):
        raise ValueError("bad shape")
    if not np.isfinite(q).all() or not np.isfinite(vecs).all():
        raise ValueError("non-finite")
    if np.linalg.norm(q) == 0 or any(np.linalg.norm(row) == 0 for row in vecs):
        raise ValueError("zero vector")
    scored = []
    for i in range(len(ids)):
        cos = float(vecs[i] @ q / (np.linalg.norm(vecs[i]) * np.linalg.norm(q)))
        scored.append((-round(cos, 9), ids[i], i))
    scored.sort()
    chosen = []
    used = 0
    for _, _, i in scored[:k]:
        cost = len(texts[i].split())
        if used + cost <= word_budget:
            chosen.append(i)
            used += cost
    lines = ["Context:"] + [f"[{ids[i]}] {texts[i]}" for i in chosen] + ["", f"Question: {query}"]
    return "\n".join(lines)
