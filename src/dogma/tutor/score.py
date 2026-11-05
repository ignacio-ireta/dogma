"""Score mode: advisory rubric scores with evidence quotes (ARCHITECTURE §4.5, ADR-0007).

The model scores a learner's written answer on a rubric. Every score must name a criterion of
the rubric, a whole number from 1 to 4, and a quote copied exactly from the learner's text. A
score that breaks one of those rules is dropped and counted. The result is advisory: it never
changes the learner's self-scores or their mastery, and the module prints no free text from the
model. The learner sees numbers and their own words.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dogma.content.rubrics import Rubric
from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.gateway import CallMeta, Gateway
from dogma.tutor.models import AdvisoryScore, AdvisoryScores, CallMode
from dogma.tutor.prompts import render_prompt
from dogma.tutor.replies import json_object

SCORE_MAX_TOKENS = 2400  # a 30-item mock with short quotes fits well inside this
MAX_QUOTE_CHARS = 300
_ITEM_KEYS = frozenset({"id", "score", "quote"})


def score_answer(gateway: Gateway, rubric: Rubric, text: str) -> AdvisoryScores:
    """Ask the model to score ``text`` on ``rubric``; keep only the scores that pass the checks."""
    prompt = render_prompt(CallMode.SCORE, None, build_user_text(rubric, text))
    reply = gateway.complete(
        CallMeta(mode=CallMode.SCORE, prompt=prompt, max_tokens=SCORE_MAX_TOKENS)
    )
    scores, dropped = parse_scores(reply.text, criteria=rubric.criteria_ids, learner_text=text)
    return AdvisoryScores(
        scores=scores,
        dropped=dropped,
        model=reply.model,
        prompt_version=prompt.version,
        latency_ms=reply.latency_ms,
    )


def build_user_text(rubric: Rubric, text: str) -> str:
    """The task, the rubric with its anchors, and the learner's text, each criterion once."""
    lines = [
        "TASK: score the LEARNER TEXT on the RUBRIC. Score only the criteria listed, 1 to 4, and "
        "give each one an exact quote from the LEARNER TEXT.",
        "",
        f"### RUBRIC ({rubric.id} v{rubric.version})",
    ]
    if rubric.kind == "mock":
        lines.append("Anchors, the same for every item:")
        lines.extend(f"{level}: {anchor}" for level, anchor in sorted(rubric.scale.anchors.items()))
        for part in rubric.parts:
            lines.append(f"{part.title}:")
            lines.extend(f"- {item.id}: {item.text}" for item in part.items)
    else:
        for dimension in rubric.dimensions:
            lines.append(f"- {dimension.id}: {dimension.question}")
            lines.extend(
                f"  {level}: {anchor}" for level, anchor in sorted(dimension.anchors.items())
            )
    lines.extend(["", "### LEARNER TEXT", text])
    return "\n".join(lines)


def parse_scores(
    reply: str, *, criteria: Sequence[str], learner_text: str
) -> tuple[tuple[AdvisoryScore, ...], int]:
    """The scores that pass every check, and how many were dropped. A bad reply is an error."""
    data = json_object(
        reply,
        ErrorCode.TUTOR_SCORE_REPLY_INVALID,
        "the model's reply was not the scores object the prompt asks for",
    )
    entries = data.get("scores")
    if not isinstance(entries, list):
        raise DogmaError(
            ErrorCode.TUTOR_SCORE_REPLY_INVALID, "the model's reply had no scores list"
        )
    allowed = frozenset(criteria)
    kept: dict[str, AdvisoryScore] = {}
    dropped = 0
    for entry in entries:
        item = _checked(entry, allowed, learner_text, kept)
        if item is None:
            dropped += 1
        else:
            kept[item.criterion] = item
    return tuple(kept.values()), dropped


def _checked(
    entry: object,
    allowed: frozenset[str],
    learner_text: str,
    kept: Mapping[str, AdvisoryScore],
) -> AdvisoryScore | None:
    if not isinstance(entry, dict) or set(entry) != _ITEM_KEYS:
        return None
    criterion, score, quote = entry["id"], entry["score"], entry["quote"]
    if not isinstance(criterion, str) or criterion not in allowed or criterion in kept:
        return None
    if type(score) is not int or not 1 <= score <= 4:
        return None
    if not isinstance(quote, str):
        return None
    clean = quote.strip()
    if not clean or len(clean) > MAX_QUOTE_CHARS or clean not in learner_text:
        return None
    return AdvisoryScore(criterion=criterion, score=score, quote=clean)
