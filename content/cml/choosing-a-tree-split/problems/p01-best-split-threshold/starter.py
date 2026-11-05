from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike


def best_split(x: ArrayLike, y: ArrayLike, *, criterion: str = "gini") -> tuple[Any, float]:
    """Return (threshold, impurity_decrease) for the best split of one numeric feature.

    Candidates are midpoints between consecutive distinct sorted values of x; x <= t goes left.
    Decrease = I(parent) - (n_L / n) I(left) - (n_R / n) I(right), with I the Gini impurity
    ("gini") or the entropy in bits ("entropy"). Scan in increasing threshold order and replace
    the incumbent only when a later decrease is larger by more than 1e-12, so ties go to the
    smallest threshold. Returns (None, 0.0) when x has fewer than two distinct values.
    Raises ValueError for empty, non-1-D or non-finite x, a mismatched y, or an unknown criterion.
    Must not mutate x or y.
    """
    raise NotImplementedError
