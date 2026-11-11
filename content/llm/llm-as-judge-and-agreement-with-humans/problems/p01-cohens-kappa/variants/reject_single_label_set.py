from __future__ import annotations

from collections.abc import Sequence


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str], labels: Sequence[str]) -> float:
    """Return Cohen's kappa (p_o - p_e) / (1 - p_e) for two raters over a fixed label set."""
    a, b, label_list = (list(rater_a), list(rater_b), list(labels))
    if not all(isinstance(x, str) for x in a + b + label_list):
        raise TypeError("labels must be str")
    if len(label_list) == 1 or len(set(label_list)) != len(label_list):
        raise ValueError("labels must be non-empty and distinct")
    if len(a) != len(b):
        raise ValueError("rater_a and rater_b must have the same length")
    if len(a) == 0:
        raise ValueError("at least one item is required")
    allowed = set(label_list)
    if any(x not in allowed for x in a + b):
        raise ValueError("every label must belong to the label set")
    n = len(a)
    p_o = sum((1 for x, y in zip(a, b, strict=True) if x == y)) / n
    p_e = 0.0
    for label in label_list:
        p_a = a.count(label) / n
        p_b = b.count(label) / n
        p_e += p_a * p_b
    if p_e == 1.0:
        raise ValueError("kappa is undefined: expected agreement is 1 (both raters used one label)")
    return (p_o - p_e) / (1.0 - p_e)
