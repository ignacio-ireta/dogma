from __future__ import annotations

from collections.abc import Sequence


def majority_vote(answers: Sequence[str]) -> str:
    """Return the most common normalised answer, with ties going to the earliest first occurrence.

    Normalise each answer with " ".join(a.split()).lower(). Drop answers that normalise to "".
    The winner has the largest count; among equal counts, the answer whose first occurrence in
    the original list is earliest wins. Return the normalised winner. Raise TypeError when answers
    is a bare str or any element is not a str, and ValueError when no answer survives. Never
    mutate the input.
    """
    raise NotImplementedError
