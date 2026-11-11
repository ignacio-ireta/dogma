from __future__ import annotations

from collections.abc import Sequence


def _normalise(answer: str) -> str:
    # Collapse internal whitespace, trim the ends, and lowercase. Punctuation is kept on purpose.
    return " ".join(answer.split()).lower()


def majority_vote(answers: Sequence[str]) -> str:
    """Wrong on purpose: strips a trailing full stop from the returned answer."""
    if isinstance(answers, str):
        raise TypeError("answers must be a list of strings, not a bare string")
    if not all(isinstance(answer, str) for answer in answers):
        raise TypeError("every answer must be a str")

    # Keep the first position of each normalised answer, so ties can be broken by position.
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for position, answer in enumerate(answers):
        key = _normalise(answer)
        if not key:  # a blank answer casts no vote
            continue
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, position)

    if not counts:
        raise ValueError("no answer survives normalisation: the list is empty or all blank")

    # Largest count first; among equal counts, the earliest first occurrence.
    winner = min(counts, key=lambda key: (-counts[key], first_seen[key]))
    return winner.rstrip(".")
