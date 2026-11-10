from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from fractions import Fraction


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str], labels: Sequence[str]) -> float:
    for seq in (rater_a, rater_b, labels):
        for value in seq:
            if not isinstance(value, str):
                raise TypeError("labels must be str")

    if len(labels) == 0:
        raise ValueError("labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be distinct")
    if len(rater_a) != len(rater_b):
        raise ValueError("rater sequences must have the same length")
    n = len(rater_a)
    if n == 0:
        raise ValueError("rater sequences must be non-empty")

    label_set = set(labels)
    for value in list(rater_a) + list(rater_b):
        if value not in label_set:
            raise ValueError(f"label {value!r} is outside the label set")

    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)  # noqa: B905
    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    # Work in exact integers: p_e = sum(count_a * count_b) / n^2.
    expected = sum(count_a[lab] * count_b[lab] for lab in labels)
    total = n * n
    if expected == total:
        raise ValueError("kappa is undefined: expected agreement is 1 (both raters used one label)")

    p_o = Fraction(agree, n)
    p_e = Fraction(expected, total)
    return float((p_o - p_e) / (1 - p_e))
