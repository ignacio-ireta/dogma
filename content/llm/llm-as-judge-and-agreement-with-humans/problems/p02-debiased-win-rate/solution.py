from __future__ import annotations

from collections.abc import Sequence

VERDICTS = frozenset({"A", "B", "tie"})


def _x_view(verdict: str, x_position: str) -> str:
    """Translate a verdict about positions into a verdict about candidates X and Y.

    In the forward order X sits in position A, so an "A" verdict means X wins. In the reverse
    order X sits in position B, so a "B" verdict means X wins. Forgetting this swap is the most
    common mistake: it reads position bias as a preference.
    """
    if verdict == "tie":
        return "tie"
    return "X" if verdict == x_position else "Y"


def debiased_win_rate(judgments: Sequence[tuple[str, str]]) -> float:
    """Return the mean item score for candidate X over pairs judged in both orders."""
    items = list(judgments)
    if not items:
        raise ValueError("judgments must not be empty")

    total = 0.0
    for item in items:
        if not isinstance(item, tuple | list) or len(item) != 2:
            raise TypeError("each judgment must be a (forward, reverse) pair")
        forward, reverse = item
        if not (isinstance(forward, str) and isinstance(reverse, str)):
            raise TypeError("verdicts must be str")
        if forward not in VERDICTS or reverse not in VERDICTS:
            raise ValueError('verdicts must be "A", "B" or "tie"')

        # Read both verdicts as statements about X and Y, not about positions.
        f_view = _x_view(forward, "A")  # forward order: X is in position A
        r_view = _x_view(reverse, "B")  # reverse order: X is in position B
        if f_view == "X" and r_view == "X":
            total += 1.0  # X wins in both orders
        elif f_view == "Y" and r_view == "Y":
            total += 0.0  # Y wins in both orders
        else:
            total += 0.5  # tie, one tie with one win, or orders that disagree

    return total / len(items)
