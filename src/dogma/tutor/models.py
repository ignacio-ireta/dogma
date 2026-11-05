"""Typed records for the tutor and its gateway (ARCHITECTURE §4.5).

Frozen dataclasses: they cross module boundaries and never carry raw dicts (ENGINEERING §4.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from dogma.core.errors import DogmaError, ErrorCode

Role = Literal["system", "user", "assistant"]
SourceKind = Literal["lesson", "corpus", "reviewed"]


# The two tutor answers a learner reads. The API schema is built from this enum, so its docstring
# stays as it was: changing it would change the public contract. Model calls use CallMode.
class Mode(StrEnum):
    """Tutor modes. explain, probe, score and interviewer arrive with later increments."""

    HINT = "hint"
    DEBUG = "debug"


class CallMode(StrEnum):
    """Every mode a model call runs in. Score and interviewer calls never produce a TutorAnswer."""

    HINT = "hint"
    DEBUG = "debug"
    SCORE = "score"  # advisory rubric scores with quotes (I-18d)
    INTERVIEWER = "interviewer"  # picks one case reveal by number, or none (I-18d)


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    max_tokens: int
    temperature: float = 0.2


@dataclass(frozen=True)
class CompletionChunk:
    """A piece of streamed text. Token counts arrive only on the final chunk, when reported."""

    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int = 0


@dataclass(frozen=True)
class RenderedPrompt:
    """The messages sent to a model, with the version and hash that the call log records."""

    mode: CallMode
    level: int | None
    version: int
    sha256: str
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Source:
    """A citable passage in the prompt: ``tag`` is what the answer writes, e.g. ``S1``."""

    tag: str
    citation: str
    kind: SourceKind


@dataclass(frozen=True)
class TutorAnswer:
    mode: Mode
    level: int | None
    text: str
    sources: tuple[Source, ...]
    cited: bool
    model: str
    prompt_version: int
    latency_ms: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewedHint:
    """One authored hint level from the problem's ``hints.md`` (CONTENT_SPEC §6.5)."""

    level: int
    label: str
    text: str


@dataclass(frozen=True)
class ReviewedHints:
    """A problem's reviewed hints. ``note`` says why there are none: missing or broken file."""

    hints: tuple[ReviewedHint, ...] = ()
    note: str | None = None

    def get(self, level: int) -> ReviewedHint | None:
        return next((hint for hint in self.hints if hint.level == level), None)

    def up_to(self, level: int) -> tuple[ReviewedHint, ...]:
        return tuple(hint for hint in self.hints if hint.level <= level)


@dataclass(frozen=True)
class HintAnswer:
    """The result of ``hint``. ``reviewed`` is None at L4 and without a usable ``hints.md``.

    ``nudge`` is the model's answer on the learner's code. It is None when the local model did not
    answer or the nudge was held back, and ``nudge_note`` then says why in one line.
    """

    level: int
    reviewed: ReviewedHint | None
    nudge: TutorAnswer | None
    nudge_note: str | None = None


@dataclass(frozen=True)
class AdvisoryScore:
    """One criterion's model score, kept only with a quote that occurs in the learner's text."""

    criterion: str
    score: int
    quote: str


@dataclass(frozen=True)
class AdvisoryScores:
    """The model's scores for one answer. ``dropped`` counts the scores refused by the checks."""

    scores: tuple[AdvisoryScore, ...]
    dropped: int
    model: str
    prompt_version: int
    latency_ms: int = 0


@dataclass(frozen=True)
class PickedReveal:
    """The interviewer's choice: the 1-based number of a reveal, or None for "nothing on that"."""

    number: int | None
    model: str
    prompt_version: int


class ProviderError(DogmaError):
    """A model call failed. ``outcome`` is what the call log records for it."""

    def __init__(self, code: ErrorCode, outcome: str, message: str) -> None:
        super().__init__(code, message)
        self.outcome = outcome
