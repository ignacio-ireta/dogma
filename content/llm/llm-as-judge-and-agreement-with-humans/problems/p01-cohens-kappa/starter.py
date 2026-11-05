from __future__ import annotations

from collections.abc import Sequence


def cohens_kappa(rater_a: Sequence[str], rater_b: Sequence[str], labels: Sequence[str]) -> float:
    """Return Cohen's kappa (p_o - p_e) / (1 - p_e) for two raters over a fixed label set.

    rater_a and rater_b are equal-length, non-empty sequences of str; labels is a non-empty
    sequence of distinct str that contains every label used. p_o is the fraction of items the
    raters agree on, and p_e is the sum over labels of the product of the two raters' marginal
    fractions. Raise ValueError when p_e == 1 (kappa is undefined), and for length mismatches,
    empty input, labels outside the label set, or a bad label set. Raise TypeError for non-str
    labels. Never mutate inputs.
    """
    raise NotImplementedError
