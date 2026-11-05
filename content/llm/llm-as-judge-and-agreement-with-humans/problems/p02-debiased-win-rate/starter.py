from __future__ import annotations

from collections.abc import Sequence


def debiased_win_rate(judgments: Sequence[tuple[str, str]]) -> float:
    """Return the mean item score for candidate X over pairs judged in both orders.

    Each judgment is (forward, reverse): forward is the verdict with X in position A, reverse with
    X in position B. Verdicts are "A", "B" or "tie", where the letter names the winning position.
    An item scores 1.0 when X wins both orders, 0.0 when Y wins both, and 0.5 otherwise (a tie, or
    orders that disagree). Raise ValueError for an empty input or a verdict outside the three
    names, and TypeError for a malformed element or a non-str verdict. Never mutate the input.
    """
    raise NotImplementedError
