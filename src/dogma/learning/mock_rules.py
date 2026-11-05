"""Mock interview rules: part totals, the band, weak and strong parts, and reveal matching (I-18c).

Pure functions over the rubric and the learner's 1-4 scores. Nothing here reads a clock, a file or
the database, so each rule is tested on its own. Percentages are the part score over the part
maximum, so parts of different sizes compare fairly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from dogma.content.rubrics import Band, Rubric
from dogma.content.schemas import Reveal

# Words that say nothing about the topic of a question. Kept short: a reveal needs a real overlap.
_STOPWORDS: Final = frozenset(
    {
        "a", "about", "and", "are", "at", "can", "could", "did", "do", "does", "for", "from",
        "how", "i", "in", "is", "it", "me", "of", "on", "or", "our", "should", "so", "tell",
        "that", "the", "their", "them", "there", "they", "this", "to", "was", "we", "what",
        "when", "where", "which", "who", "why", "with", "would", "you", "your",
    }
)  # fmt: skip
_WORD: Final = re.compile(r"[a-z0-9]+")
_MIN_OVERLAP: Final = 2  # one shared word is luck; two is a question about the same thing


@dataclass(frozen=True)
class PartScore:
    """One rubric part's score for one mock: the sum of its items, against the part maximum."""

    part_id: str
    title: str
    minutes: str
    score: int
    maximum: int

    @property
    def percent(self) -> float:
        return self.score / self.maximum


def part_scores(rubric: Rubric, scores: Mapping[str, int]) -> tuple[PartScore, ...]:
    """Each part's total, in rubric order. ``scores`` must already be checked against the rubric."""
    return tuple(
        PartScore(
            part_id=part.id,
            title=part.title,
            minutes=part.minutes,
            score=sum(scores[item.id] for item in part.items),
            maximum=part.maximum,
        )
        for part in rubric.parts
    )


def total_of(parts: Sequence[PartScore]) -> int:
    return sum(part.score for part in parts)


def band_for(rubric: Rubric, total: int) -> Band:
    """The interpretation band that holds ``total``. Bands cover 0 to the maximum without gaps."""
    for band in rubric.bands:
        if band.low <= total <= band.high:
            return band
    raise ValueError(f"total {total} is outside the bands of {rubric.id} v{rubric.version}")


def weakest_part(parts: Sequence[PartScore]) -> PartScore:
    """The lowest percentage. A tie goes to the part earlier in the rubric."""
    return min(parts, key=lambda part: part.percent)


def strongest_part(parts: Sequence[PartScore]) -> PartScore:
    """The highest percentage. A tie goes to the part earlier in the rubric."""
    return min(parts, key=lambda part: (-part.percent, parts.index(part)))


def weakest_over(rubric: Rubric, mocks: Sequence[Mapping[str, int]]) -> PartScore | None:
    """The weakest part across several mocks, from the summed part totals. None when there are none.

    Summing before dividing means a long mock counts for more than a short one, and a part whose
    maximum is larger is not punished for its size.
    """
    if not mocks:
        return None
    per_mock = [part_scores(rubric, scores) for scores in mocks]
    combined = tuple(
        PartScore(
            part_id=part.part_id,
            title=part.title,
            minutes=part.minutes,
            score=sum(mock[index].score for mock in per_mock),
            maximum=sum(mock[index].maximum for mock in per_mock),
        )
        for index, part in enumerate(per_mock[0])
    )
    return weakest_part(combined)


def reveal_for(question: str, reveals: Sequence[Reveal]) -> Reveal | None:
    """The reveal whose ``when_asked`` shares the most keywords with ``question``, or None.

    A reveal needs at least two shared keywords, or every keyword it has when it has only one.
    Ties go to the reveal that comes first in reveals.yaml, so the answer never depends on luck.
    """
    asked = _keywords(question)
    best: Reveal | None = None
    best_overlap = 0
    for reveal in reveals:
        wanted = _keywords(reveal.when_asked)
        needed = min(_MIN_OVERLAP, len(wanted))
        overlap = len(asked & wanted)
        if wanted and overlap >= needed and overlap > best_overlap:
            best, best_overlap = reveal, overlap
    return best


def shares_keyword(question: str, when_asked: str) -> bool:
    """True when the question and a reveal's question share at least one keyword.

    This is the weaker gate used for a model's pick: the model chooses among a short list, and
    one shared content word stops a pick that has nothing to do with the question.
    """
    return bool(_keywords(question) & _keywords(when_asked))


def _keywords(text: str) -> frozenset[str]:
    words = (_singular(word) for word in _WORD.findall(text.lower()))
    return frozenset(word for word in words if word not in _STOPWORDS and len(word) > 1)


def _singular(word: str) -> str:
    """Drop a plural 's' so 'segments' and 'segment' match. Words of three letters or fewer stay."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word
