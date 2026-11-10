from __future__ import annotations

from collections.abc import Sequence

_VERDICTS = ("A", "B", "tie")


def debiased_win_rate(judgments: Sequence[tuple[str, str]]) -> float:
    """Return the mean item score for candidate X over pairs judged in both orders.

    Each judgment is (forward, reverse): forward is the verdict with X in position A, reverse with
    X in position B. Verdicts are "A", "B" or "tie", where the letter names the winning position.
    An item scores 1.0 when X wins both orders, 0.0 when Y wins both, and 0.5 otherwise (a tie, or
    orders that disagree). Raise ValueError for an empty input or a verdict outside the three
    names, and TypeError for a malformed element or a non-str verdict. Never mutate the input.
    """
    if not isinstance(judgments, (list, tuple)):
        raise TypeError("judgments must be a list or tuple")
    if len(judgments) == 0:
        raise ValueError("judgments must not be empty")

    total = 0.0
    for item in judgments:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError("each judgment must be a 2-item tuple or list (forward, reverse)")
        forward, reverse = item
        for verdict in (forward, reverse):
            if not isinstance(verdict, str):
                raise TypeError("verdicts must be str")
            if verdict not in _VERDICTS:
                raise ValueError(f"verdict must be one of 'A', 'B', 'tie'; got {verdict!r}")

    for forward, reverse in judgments:
        x_wins_forward = forward == "A"
        y_wins_forward = forward == "B"
        x_wins_reverse = reverse == "B"
        y_wins_reverse = reverse == "A"
        if x_wins_forward and x_wins_reverse:
            total += 1.0
        elif y_wins_forward and y_wins_reverse:
            total += 0.0
        else:
            total += 0.5

    return float(total / len(judgments))
