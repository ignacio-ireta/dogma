from __future__ import annotations


def majority_vote(answers):
    if isinstance(answers, str):
        raise TypeError("bare string")
    if not all(type(a) is str for a in answers):
        raise TypeError("non-str element")
    if len(answers) == 0:
        raise ValueError("empty")
    counts = {}
    first = {}
    for pos, a in enumerate(answers):
        k = " ".join(a.split()).lower()
        if k == "":
            continue
        counts[k] = counts.get(k, 0) + 1
        if k not in first:
            first[k] = pos
    if not counts:
        raise ValueError("all blank")
    best = None
    for k in counts:  # noqa: PLC0206
        key = (-counts[k], first[k])
        if best is None or key < best[0]:
            best = (key, k)
    return best[1]
