"""``dogma answer``: answer a topic's interview question, score it, and see the mastery it earns.

The question is shown by its title only, so the 60-second answer is not on screen while the learner
answers. The answer comes from ``--file``, from ``--spoken`` (answered aloud, no text kept) or from
an interactive prompt that ends at an empty line. The four scores come from ``--scores`` or from
prompts. Each dimension's anchors are printed first, so the score means what the rubric says.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from dogma.cli.context import CliError, out, settings_of
from dogma.cli.model_view import DEGRADABLE, print_advisory, unavailable_note
from dogma.cli.tutor import make_gateway
from dogma.content.interview import load_question
from dogma.content.rubrics import Rubric, load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import DogmaError, ErrorCode
from dogma.db.models import Mastery
from dogma.learning.advisory import attach_to_answer, model_score_payload
from dogma.learning.answers import (
    TOPIC_ANSWER_ID,
    TOPIC_ANSWER_VERSION,
    AnswerResult,
    record_answer,
)
from dogma.tutor.score import score_answer

TopicArg = Annotated[str, typer.Argument(help="Topic id, e.g. cml/vector-operations.")]
QuestionOpt = Annotated[
    int, typer.Option("--question", min=1, help="Question number in interview.md.")
]
FileOpt = Annotated[
    Path | None,
    typer.Option("--file", help="Read the written answer from this text file."),
]
SpokenOpt = Annotated[
    bool, typer.Option("--spoken", help="You answered aloud: record the scores, keep no text.")
]
ScoresOpt = Annotated[
    str | None,
    typer.Option("--scores", help="Scores as accuracy=3,structure=4,trade-offs=2,communication=3."),
]
NoteOpt = Annotated[str | None, typer.Option("--note", help="An optional note for yourself.")]
ModelScoreOpt = Annotated[
    bool,
    typer.Option(
        "--model-score",
        help="Also ask the local model for advisory scores, with quotes from your text. Your "
        "scores stay the source of truth.",
    ),
]


def answer_command(
    ctx: typer.Context,
    *,
    topic: TopicArg,
    question: QuestionOpt = 1,
    file_path: FileOpt = None,
    spoken: SpokenOpt = False,
    scores: ScoresOpt = None,
    note: NoteOpt = None,
    model_score: ModelScoreOpt = False,
) -> None:
    """Answer a topic's interview question, score it on the topic-answer rubric, and store it."""
    settings = settings_of(ctx)
    if not settings.record_attempts:
        raise DogmaError(
            ErrorCode.ANSWER_RECORDING_OFF,
            "Recording is switched off (DOGMA_RECORD_ATTEMPTS=false), so nothing was stored.",
        )
    if file_path is not None and spoken:
        raise CliError("choose --file or --spoken, not both.")
    console = out()
    item = load_question(settings.content_dir / topic, question)
    console.print(
        f"[bold]{escape(topic)} · Q{item.number}[/]\n{escape(item.title)}", soft_wrap=True
    )
    text = None if spoken else _read_answer(console, file_path)
    rubric = load_rubric_version(settings.curriculum_dir, TOPIC_ANSWER_ID, TOPIC_ANSWER_VERSION)
    _print_anchors(console, rubric)
    chosen = parse_scores(scores) if scores is not None else _prompt_scores(console, rubric)
    result = record_answer(
        settings,
        topic_id=topic,
        prompt_ref=f"interview.md#Q{item.number}",
        text=text,
        scores=chosen,
        notes=note,
        now=datetime.now(UTC),
    )
    _print_result(console, result, rubric)
    if model_score:
        _model_section(settings, console, result=result, text=text, rubric=rubric, scores=chosen)


def _model_section(
    settings: Settings,
    console: Console,
    *,
    result: AnswerResult,
    text: str | None,
    rubric: Rubric,
    scores: dict[str, int],
) -> None:
    """Ask the model for advisory scores on the stored text. A down model only adds a note."""
    if text is None:
        console.print(
            "[dim]Given aloud, so there is no text for the model to score. Your scores are "
            "stored.[/]",
            soft_wrap=True,
        )
        return
    try:
        advisory = score_answer(make_gateway(settings), rubric, text.strip())
    except DogmaError as exc:
        if exc.code not in DEGRADABLE:
            raise
        unavailable_note(console, exc)
        return
    attach_to_answer(
        settings,
        result.answer_id,
        model_score_payload(advisory, rubric_id=rubric.id, rubric_version=rubric.version),
    )
    print_advisory(console, self_scores=scores, advisory=advisory)


def parse_scores(text: str) -> dict[str, int]:
    """``accuracy=3,structure=4`` as a dict. Range and key checks belong to the service."""
    scores: dict[str, int] = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            raise CliError(f"score {part.strip()!r} should look like accuracy=3")
        if key in scores:
            raise CliError(f"{key} is scored twice")
        try:
            scores[key] = int(value)
        except ValueError as exc:
            raise CliError(f"{key} must be a whole number from 1 to 4, not {value!r}") from exc
    return scores


def _read_answer(console: Console, file_path: Path | None) -> str:
    if file_path is not None:
        if not file_path.is_file():
            raise CliError(f"no file at {file_path}")
        return file_path.read_text(encoding="utf-8")
    console.print("Type your answer. Finish with an empty line.", soft_wrap=True)
    lines: list[str] = []
    while True:
        line = _input(console)
        if line is None or not line.strip():
            break
        lines.append(line)
    if not lines:
        raise CliError("no answer given: use --file PATH, --spoken, or type one.")
    return "\n".join(lines)


def _input(console: Console) -> str | None:
    try:
        return console.input("")
    except (EOFError, KeyboardInterrupt):
        return None


def _print_anchors(console: Console, rubric: Rubric) -> None:
    console.print(
        f"\n[bold]Scoring[/] ({rubric.id} v{rubric.version}): 1 to 4 on each dimension.",
        soft_wrap=True,
    )
    for dimension in rubric.dimensions:
        console.print(
            f"[bold]{escape(dimension.title)}[/] · {escape(dimension.question)}", soft_wrap=True
        )
        for level in range(rubric.scale.low, rubric.scale.high + 1):
            console.print(f"  {level}  {escape(dimension.anchors[level])}", soft_wrap=True)


def _prompt_scores(console: Console, rubric: Rubric) -> dict[str, int]:
    chosen: dict[str, int] = {}
    for dimension in rubric.dimensions:
        while True:
            raw = _input_prompt(console, f"{dimension.title} (1-4)")
            if raw is None:
                raise CliError("the scores were not finished; nothing was stored.")
            try:
                value = int(raw.strip())
            except ValueError:
                value = 0
            if rubric.scale.low <= value <= rubric.scale.high:
                chosen[dimension.id] = value
                break
            console.print("Choose a whole number from 1 to 4.")
    return chosen


def _input_prompt(console: Console, message: str) -> str | None:
    try:
        return console.input(f"{message}: ")
    except (EOFError, KeyboardInterrupt):
        return None


def _print_result(console: Console, result: AnswerResult, rubric: Rubric) -> None:
    how = "answered aloud" if result.spoken else "written"
    shown = " · ".join(
        f"{dimension.title.lower()} {result.scores[dimension.id]}"
        for dimension in rubric.dimensions
    )
    console.print(
        f"\nStored answer {result.answer_id[:8]} for {escape(result.topic_id)} "
        f"({result.prompt_ref}, {how}).",
        soft_wrap=True,
    )
    console.print(f"Scores: {escape(shown)}", soft_wrap=True)
    if result.promoted:
        console.print(f"Mastery: {result.before.value} -> {result.after.value}", soft_wrap=True)
    else:
        console.print(f"Mastery: {_label(result.after)} (unchanged)", soft_wrap=True)
    console.print(f"Why: {escape(result.reason)}", soft_wrap=True)


def _label(mastery: Mastery) -> str:
    return "not started" if mastery is Mastery.UNSEEN else mastery.value
