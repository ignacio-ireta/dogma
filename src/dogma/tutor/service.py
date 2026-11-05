"""The two tutor use cases, hint and debug, plus the output guards (ARCHITECTURE §4.5, LEARNER §7).

The guards are deterministic and run on every answer: judgements about the learner's code are
removed, long code in a debug answer is omitted, and an answer with no valid source citation is
marked unsourced. The walkthrough (L4) is locked until two recorded submits of the problem have
failed, unless the learner confirms it.

A hint at levels 1 to 3 (I-22) starts from the problem's reviewed ``hints.md``. The reviewed hint
for the level is ready before any model call, so the CLI can show it at once. Then the model writes
a nudge on the learner's code, with the reviewed hints up to that level as a ceiling. The ceiling
guard removes formulas that only a later level gives. When the local model does not answer, the
nudge is skipped with a one-line note, and the call still succeeds.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dogma.content.schemas import ProblemMeta
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.grading.models import GradeReport, RunStatus
from dogma.tutor.ceiling import FORMULA_NOTE, guard_ceiling
from dogma.tutor.context import AssembledPrompt, assemble_prompt
from dogma.tutor.evidence import (
    FAILED_STATUSES,
    UNLOCK_FAILED_SUBMITS,
    Evidence,
    gather_evidence,
)
from dogma.tutor.gateway import CallMeta, Gateway
from dogma.tutor.hints import REVIEWED_LEVELS
from dogma.tutor.models import (
    CallMode,
    CompletionResult,
    HintAnswer,
    Mode,
    ReviewedHint,
    Source,
    TutorAnswer,
)

HINT_MAX_TOKENS = 700
DEBUG_MAX_TOKENS = 500
DEBUG_MAX_CODE_LINES = 3
_DEGRADABLE = frozenset({ErrorCode.LLM_UNAVAILABLE, ErrorCode.LLM_RESPONSE_INVALID})
_CODE_BLOCK = re.compile(r"^(```|~~~)[^\n]*\n(.*?)\n\1[ \t]*$", re.MULTILINE | re.DOTALL)
_CITATION = re.compile(r"\[(S\d+)\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_SUBJECT = (
    r"(?:your|the|this|that|my)\s+(?:\w+\s+){0,2}?"
    r"(?:code|solution|implementation|function|answer|submission|attempt|program|script)"
)
_JUDGEMENTS = (
    re.compile(
        rf"\b{_SUBJECT}\b[^.!?\n]{{0,40}}\b(?:is|are|looks|seems|was|works)\s+(?:not\s+)?"
        r"(?:correct|incorrect|wrong|right|valid|invalid|fine|broken)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:your|the)\s+(?:code|solution|implementation|function)\s+(?:passes|fails)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:got|have)\s+it\s+(?:right|wrong)\b", re.IGNORECASE),
)
_SHORT_VERDICT = re.compile(
    r"\b(?:this|that|it)\s+(?:is|looks)\s+(?:correct|incorrect|wrong|right)\b", re.IGNORECASE
)
_GRADER_NOTE = "(Whether the code is correct is the grader's call: run dogma submit.)"
_UNSOURCED_NOTE = "Unsourced: no lesson or corpus source backs this answer."
_OMITTED_CODE = "[code omitted: debug names the line and the error; it does not rewrite the code]"
_HELD_BACK = (
    "The nudge was held back: nothing in it stayed within the reviewed hint. "
    "The reviewed hint above stands."
)


@dataclass(frozen=True)
class Target:
    """The problem the learner named, already resolved from content/."""

    problem_dir: Path
    problem_id: str
    meta: ProblemMeta


@dataclass(frozen=True)
class PreparedHint:
    """A hint call whose reviewed part is known. The model has not been called yet.

    ``reviewed`` is the authored hint for the level when the problem's hints.md is usable and the
    level is 1 to 3. ``hints_note`` says why there is none, at those levels. ``ceiling`` holds the
    reviewed texts up to the level and ``above`` the texts after it: the second is only checked,
    never sent to the model.
    """

    settings: Settings
    level: int
    problem_id: str
    reviewed: ReviewedHint | None
    hints_note: str | None
    assembled: AssembledPrompt
    ceiling: tuple[str, ...]
    above: tuple[str, ...]
    problem_text: str

    def complete(self, gateway: Gateway | None = None) -> HintAnswer:
        """Call the model. Without a reviewed hint this is the I-12 model-only answer."""
        active = gateway if gateway is not None else Gateway(self.settings)
        try:
            reply = active.complete(
                CallMeta(
                    mode=CallMode.HINT,
                    prompt=self.assembled.prompt,
                    max_tokens=HINT_MAX_TOKENS,
                    problem_id=self.problem_id,
                )
            )
        except DogmaError as exc:
            if self.reviewed is None or exc.code not in _DEGRADABLE:
                raise
            return HintAnswer(
                level=self.level,
                reviewed=self.reviewed,
                nudge=None,
                nudge_note=f"Personalised nudge needs the local model: {exc.message}",
            )
        text, _ = guard_verdicts(reply.text.strip())
        if self.reviewed is not None:
            body, removed = guard_ceiling(
                text, shown=self.ceiling, above=self.above, problem=self.problem_text
            )
            if not body:
                return HintAnswer(
                    level=self.level, reviewed=self.reviewed, nudge=None, nudge_note=_HELD_BACK
                )
            text = f"{body}\n{FORMULA_NOTE}" if removed else body
        nudge = _finish(reply, self.assembled, mode=Mode.HINT, level=self.level, text=text)
        return HintAnswer(level=self.level, reviewed=self.reviewed, nudge=nudge)


def prepare_hint(
    settings: Settings,
    target: Target,
    *,
    level: int,
    yes: bool,
    ask: Callable[[str], bool],
) -> PreparedHint:
    """Everything a hint needs before the model call: the L4 gate, evidence and reviewed hint."""
    evidence = _evidence(settings, target)
    if (
        level == 4
        and not _walkthrough_open(evidence)
        and not (yes or ask(_lock_question(evidence)))
    ):
        raise DogmaError(
            ErrorCode.TUTOR_LOCKED,
            f"the walkthrough (L4) is locked until {UNLOCK_FAILED_SUBMITS} recorded submits of "
            f"this problem fail (you have {evidence.history.failed_submits}); "
            "confirm to open it early (CLI: --yes; API: confirm_walkthrough)",
        )
    reviewed = evidence.reviewed
    grounded = level in REVIEWED_LEVELS and bool(reviewed.hints)
    return PreparedHint(
        settings=settings,
        level=level,
        problem_id=target.problem_id,
        reviewed=reviewed.get(level) if grounded else None,
        hints_note=reviewed.note if level in REVIEWED_LEVELS and not grounded else None,
        assembled=assemble_prompt(evidence, mode=CallMode.HINT, level=level),
        ceiling=tuple(hint.text for hint in reviewed.up_to(level)) if grounded else (),
        above=tuple(hint.text for hint in reviewed.hints if hint.level > level) if grounded else (),
        problem_text=evidence.problem_text,
    )


def hint(
    settings: Settings,
    target: Target,
    *,
    level: int,
    yes: bool,
    ask: Callable[[str], bool],
    gateway: Gateway | None = None,
) -> HintAnswer:
    """One hint at ladder level 1 to 4. Level 4 asks first unless unlocked or ``yes``."""
    return prepare_hint(settings, target, level=level, yes=yes, ask=ask).complete(gateway)


def debug(settings: Settings, target: Target, *, gateway: Gateway | None = None) -> TutorAnswer:
    """Name the first wrong concept or line of the latest recorded failing attempt."""
    evidence = _evidence(settings, target)
    latest = evidence.history.latest
    if not evidence.history.known:
        raise DogmaError(
            ErrorCode.TUTOR_NO_FAILURE,
            "the learner database cannot be read, so no failing attempt can be found",
        )
    if latest is None or latest.status not in FAILED_STATUSES:
        raise DogmaError(ErrorCode.TUTOR_NO_FAILURE, _no_failure_message(latest))
    assembled = assemble_prompt(evidence, mode=CallMode.DEBUG)
    active = gateway if gateway is not None else Gateway(settings)
    reply = active.complete(
        CallMeta(
            mode=CallMode.DEBUG,
            prompt=assembled.prompt,
            max_tokens=DEBUG_MAX_TOKENS,
            problem_id=target.problem_id,
        )
    )
    text, _ = guard_verdicts(reply.text.strip())
    text, _ = omit_long_code(text)
    return _finish(reply, assembled, mode=Mode.DEBUG, level=None, text=text)


def guard_verdicts(text: str) -> tuple[str, bool]:
    """Drop sentences that call the learner's code correct, incorrect, passing or failing."""
    changed = False
    lines: list[str] = []
    for line in text.split("\n"):
        kept: list[str] = []
        for sentence in _SENTENCE_END.split(line):
            if _is_judgement(sentence):
                changed = True
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    guarded = "\n".join(lines)
    if changed:
        guarded = f"{guarded}\n{_GRADER_NOTE}"
    return guarded, changed


def omit_long_code(text: str) -> tuple[str, bool]:
    """Replace fenced code longer than a few lines. Debug names lines; it does not rewrite code."""
    changed = False

    def keep_or_omit(match: re.Match[str]) -> str:
        nonlocal changed
        if match.group(2).count("\n") + 1 <= DEBUG_MAX_CODE_LINES:
            return match.group(0)
        changed = True
        return _OMITTED_CODE

    return _CODE_BLOCK.sub(keep_or_omit, text), changed


def cites(text: str, sources: tuple[Source, ...]) -> bool:
    """True when the text cites at least one source tag that this call actually supplied."""
    tags = set(_CITATION.findall(text))
    return any(source.tag in tags for source in sources)


def _evidence(settings: Settings, target: Target) -> Evidence:
    return gather_evidence(
        settings,
        problem_dir=target.problem_dir,
        problem_id=target.problem_id,
        meta=target.meta,
    )


def _walkthrough_open(evidence: Evidence) -> bool:
    return evidence.history.known and evidence.history.failed_submits >= UNLOCK_FAILED_SUBMITS


def _lock_question(evidence: Evidence) -> str:
    return (
        f"The walkthrough (L4) is locked: {evidence.history.failed_submits} of "
        f"{UNLOCK_FAILED_SUBMITS} recorded submits that did not pass. Open it now?"
    )


def _no_failure_message(latest: GradeReport | None) -> str:
    if latest is None:
        return "no recorded attempt for this problem yet: run dogma submit first"
    if latest.status is RunStatus.PASSED:
        return "your latest recorded attempt passed, so there is nothing to debug"
    if latest.status is RunStatus.REJECTED:
        return "your latest attempt was rejected before it ran: dogma submit lists the findings"
    return "no failing recorded attempt for this problem: run dogma submit first"


def _finish(
    reply: CompletionResult,
    assembled: AssembledPrompt,
    *,
    mode: Mode,
    level: int | None,
    text: str,
) -> TutorAnswer:
    """Mark an answer unsourced when no supplied source is cited, then wrap it with its metadata."""
    cited = cites(text, assembled.sources)
    if not cited and "unsourced" not in text.lower():
        text = f"{text}\n\n{_UNSOURCED_NOTE}"
    return TutorAnswer(
        mode=mode,
        level=level,
        text=text.strip(),
        sources=assembled.sources,
        cited=cited,
        model=reply.model,
        prompt_version=assembled.prompt.version,
        latency_ms=reply.latency_ms,
        notes=assembled.notes,
    )


def _is_judgement(sentence: str) -> bool:
    if any(pattern.search(sentence) for pattern in _JUDGEMENTS):
        return True
    return len(sentence.split()) <= 5 and _SHORT_VERDICT.search(sentence) is not None
