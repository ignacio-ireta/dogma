"""Terminal view of the model's advisory scores, and the errors that mean "no model answered".

The model's own words are never printed. A line shows the criterion, the learner's score, the
model's score and the learner's own quote, then one summary line. A down or unusable model is a
note, and the command still exits 0 with the self-scores stored.
"""

from __future__ import annotations

from collections.abc import Mapping

from rich.console import Console
from rich.markup import escape

from dogma.core.errors import DogmaError, ErrorCode
from dogma.learning.advisory import compare
from dogma.tutor.models import AdvisoryScores

# Errors that mean the model did not give a usable answer. The self-scores are stored anyway.
DEGRADABLE = frozenset(
    {
        ErrorCode.LLM_UNAVAILABLE,
        ErrorCode.LLM_RESPONSE_INVALID,
        ErrorCode.TUTOR_SCORE_REPLY_INVALID,
        ErrorCode.TUTOR_INTERVIEWER_REPLY_INVALID,
    }
)


def print_advisory(
    console: Console, *, self_scores: Mapping[str, int], advisory: AdvisoryScores
) -> None:
    """Each scored criterion beside the learner's score, its quote, and the one-line comparison."""
    console.print("\n[bold]Model scores (advisory)[/]", soft_wrap=True)
    for item in advisory.scores:
        own = self_scores.get(item.criterion)
        own_text = "not scored" if own is None else str(own)
        console.print(
            f"  {escape(item.criterion)}: you {own_text}, model {item.score} "
            f'· "{escape(item.quote)}"',
            soft_wrap=True,
        )
    if advisory.dropped:
        console.print(
            f"[dim]{advisory.dropped} model score(s) were dropped: a quote was not in your "
            "text, or a rule was broken.[/]",
            soft_wrap=True,
        )
    console.print(escape(compare(self_scores, advisory).summary), soft_wrap=True)
    console.print(
        f"[dim]{escape(advisory.model)} · prompt score v{advisory.prompt_version} · "
        "advisory only: it does not change your score or your mastery.[/]",
        soft_wrap=True,
    )


def unavailable_note(console: Console, exc: DogmaError) -> None:
    """One dim line: the model did not answer. The self-scores are already stored."""
    message = exc.message.rstrip(".")
    console.print(
        f"[dim]{escape(f'Model score unavailable: {message}. Your self-scores are stored.')}[/]",
        soft_wrap=True,
    )
