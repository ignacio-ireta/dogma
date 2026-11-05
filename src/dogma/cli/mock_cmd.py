"""``dogma mock``: a timed mock interview with a case, a coding problem and a theory question.

``start`` plans the mock and shows the part timings. ``reveal`` answers a question about the case
only when the question matches a reveal. ``score`` takes the 30 item scores from a JSON file or asks
for each one. ``debrief`` shows the last scored mock. ``abandon`` closes the open mock without
scores. ``status`` shows what is open. Every command is wrapped by ``friendly_errors``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from dogma.cli.context import CliError, friendly_errors, out, settings_of
from dogma.cli.model_view import DEGRADABLE, print_advisory, unavailable_note
from dogma.cli.tutor import make_gateway
from dogma.content.rubrics import Rubric, load_rubric_version
from dogma.core.config import Settings
from dogma.core.errors import DogmaError
from dogma.learning.advisory import attach_to_mock, mock_answer_text, model_score_payload
from dogma.learning.mock_rules import PartScore
from dogma.learning.mocks import (
    MOCK_RUBRIC_ID,
    MOCK_RUBRIC_VERSION,
    MockDebrief,
    MockPlan,
    MockReport,
    RevealResult,
    abandon_mock,
    ask_interviewer,
    debrief,
    keyword_reveal,
    mock_status,
    open_case_material,
    reveal,
    score_mock,
    start_mock,
)
from dogma.tutor.score import score_answer

app = typer.Typer(
    help="Mock interviews: start, reveal, ask, score, debrief, abandon, status.",
    epilog="Example: dogma mock start",
    no_args_is_help=True,
    add_completion=False,
)

CaseOpt = Annotated[
    str | None,
    typer.Option(
        "--case", help="A sys-track case id. Default: the one chosen by the planner rules."
    ),
]
ScoresFileOpt = Annotated[
    Path | None,
    typer.Option(
        "--scores-file",
        help="JSON object mapping each item id to a score from 1 to 4. Default: ask for each item.",
    ),
]
QuestionArg = Annotated[str, typer.Argument(help="What you asked the interviewer, in your words.")]
ModelScoreOpt = Annotated[
    bool,
    typer.Option(
        "--model-score",
        help="Also ask the local model for advisory scores. It reads your item notes only; "
        "your scores stay the source of truth.",
    ),
]


def start_command(ctx: typer.Context, *, case: CaseOpt = None) -> None:
    """Start a mock: pick the case, coding problem and theory question, and show the timings."""
    settings = settings_of(ctx)
    plan = start_mock(settings, now=datetime.now(UTC), case_id=case)
    _print_plan(out(), plan)


def reveal_command(ctx: typer.Context, *, question: QuestionArg) -> None:
    """Ask the interviewer something about the case. You get a fact only for a matching question."""
    settings = settings_of(ctx)
    result = reveal(settings, question)
    console = out()
    console.print(escape(result.text), soft_wrap=True)
    if result.exhibit is not None:
        console.print(f"[dim]Exhibit: {escape(result.exhibit)}[/]", soft_wrap=True)


def score_command(
    ctx: typer.Context, *, scores_file: ScoresFileOpt = None, model_score: ModelScoreOpt = False
) -> None:
    """Score the 30 items of the open mock, from a file or one by one, and store the result."""
    settings = settings_of(ctx)
    console = out()
    rubric = _rubric(settings)
    if scores_file is not None:
        scores = _read_scores(scores_file)
        notes: dict[str, str] = {}
    else:
        scores, notes = _ask_items(console, rubric)
    report = score_mock(settings, scores, now=datetime.now(UTC), notes=notes)
    _print_report(console, report)
    console.print(
        "Stored. Run dogma mock debrief for the part breakdown, then dogma close.",
        soft_wrap=True,
    )
    if model_score:
        _model_section(
            settings,
            console,
            session_id=report.session_id,
            rubric=rubric,
            scores={key: value for key, value in scores.items() if isinstance(value, int)},
        )


def _model_section(
    settings: Settings,
    console: Console,
    *,
    session_id: str,
    rubric: Rubric,
    scores: dict[str, int],
) -> None:
    """Advisory scores from the mock's item notes. A down model only adds a note."""
    text = mock_answer_text(settings, session_id)
    if not text:
        console.print(
            "[dim]No item notes to score: the model reads your notes only, and you wrote none. "
            "Your scores are stored.[/]",
            soft_wrap=True,
        )
        return
    try:
        advisory = score_answer(make_gateway(settings), rubric, text)
    except DogmaError as exc:
        if exc.code not in DEGRADABLE:
            raise
        unavailable_note(console, exc)
        return
    attach_to_mock(
        settings,
        session_id,
        model_score_payload(advisory, rubric_id=rubric.id, rubric_version=rubric.version),
    )
    print_advisory(console, self_scores=scores, advisory=advisory)


def debrief_command(ctx: typer.Context) -> None:
    """Show the last scored mock: part totals, the band, strength, weakness and case debrief."""
    view = debrief(settings_of(ctx))
    _print_debrief(out(), view)


def abandon_command(ctx: typer.Context) -> None:
    """Close the open mock without scores. Nothing is stored for it."""
    abandon_mock(settings_of(ctx), now=datetime.now(UTC))
    out().print("Abandoned the open mock. No scores were stored.", soft_wrap=True)


def status_command(ctx: typer.Context) -> None:
    """Show the open mock, if there is one, and the last scored mock."""
    view = mock_status(settings_of(ctx))
    console = out()
    if view.open is None:
        console.print("No mock is open. Start one with dogma mock start.", soft_wrap=True)
    else:
        console.print("A mock is open.", soft_wrap=True)
        _print_plan(console, view.open)
    if view.last_scored is not None:
        console.print(
            f"Last scored mock: {view.last_scored.total} of {view.last_scored.maximum} "
            f"({escape(view.last_scored.band.label)}). Run dogma mock debrief for detail.",
            soft_wrap=True,
        )


def ask_command(ctx: typer.Context, *, question: QuestionArg) -> None:
    """Ask the interviewer. The local model releases one matching reveal, or says there is none."""
    settings = settings_of(ctx)
    console = out()
    try:
        result = ask_interviewer(settings, make_gateway(settings), question)
        note = None
    except DogmaError as exc:
        if exc.code not in DEGRADABLE:
            raise
        result = keyword_reveal(open_case_material(settings), question)
        note = f"The local model could not answer ({exc.message}), so the keyword match answered."
    _print_reveal(console, result)
    if result.via == "model" and result.matched:
        console.print("[dim]Chosen by the local model. The words come from the case file.[/]")
    if note is not None:
        console.print(f"[dim]{escape(note)}[/]", soft_wrap=True)


def _print_reveal(console: Console, result: RevealResult) -> None:
    console.print(escape(result.text), soft_wrap=True)
    if result.exhibit is not None:
        console.print(f"[dim]Exhibit: {escape(result.exhibit)}[/]", soft_wrap=True)


def _rubric(settings: Settings) -> Rubric:
    return load_rubric_version(settings.curriculum_dir, MOCK_RUBRIC_ID, MOCK_RUBRIC_VERSION)


def _read_scores(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise CliError(f"no scores file at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CliError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CliError(f"{path} should hold one JSON object mapping item ids to scores")
    return dict(data)


def _ask_items(console: Console, rubric: Rubric) -> tuple[dict[str, object], dict[str, str]]:
    """Ask every item in rubric order. A score must be 1 to 4; a note is optional."""
    console.print(
        "Score each item from 1 (did not show it) to 4 (exceptional). "
        "Press Enter after a score to skip its note.",
        soft_wrap=True,
    )
    scores: dict[str, object] = {}
    notes: dict[str, str] = {}
    for part in rubric.parts:
        console.print(f"\n[bold]{escape(part.title)}[/]")
        for item in part.items:
            scores[item.id] = _ask_score(console, item.text)
            note = _ask(console, "  Note (Enter to skip)")
            if note is None:
                raise CliError("the scores were not finished; nothing was stored.")
            if note.strip():
                notes[item.id] = note.strip()
    return scores, notes


def _ask_score(console: Console, text: str) -> object:
    while True:
        raw = _ask(console, f"{escape(text)} (1-4)")
        if raw is None:
            raise CliError("the scores were not finished; nothing was stored.")
        try:
            value = int(raw.strip())
        except ValueError:
            console.print("Choose a whole number from 1 to 4.")
            continue
        if 1 <= value <= 4:
            return value
        console.print("Choose a whole number from 1 to 4.")


def _ask(console: Console, message: str) -> str | None:
    try:
        return console.input(f"{message}: ")
    except (EOFError, KeyboardInterrupt):
        return None


def _print_plan(console: Console, plan: MockPlan) -> None:
    console.print(
        f"[bold]Mock started[/] · {plan.started_at.astimezone():%a %d %b %Y, %H:%M}",
        soft_wrap=True,
    )
    console.print(
        f"Case: {_named(plan.case_title, plan.case_id)}",
        soft_wrap=True,
    )
    coding = _named(plan.coding_title, plan.coding_problem_id, "none yet, pass a core problem")
    console.print(f"Coding: {coding}", soft_wrap=True)
    theory = _named(plan.theory_title, plan.theory_topic_id, "none yet, retain a topic")
    if plan.theory_question_title is not None:
        theory = f"{theory}, question {plan.theory_question}: {plan.theory_question_title}"
    console.print(f"Theory: {theory}", soft_wrap=True)
    console.print("Timing:")
    for title, minutes in plan.part_minutes:
        shown = minutes if minutes == "throughout" else f"{minutes} min"
        console.print(f"  {escape(title)}: {shown}", soft_wrap=True)
    if plan.focus is not None:
        console.print(
            f"Focus: {escape(plan.focus.title)} was your weakest part over the last scored mocks "
            f"({round(plan.focus.percent * 100)}%).",
            soft_wrap=True,
        )
    console.print(
        'Next: dogma mock reveal "<your question>" during the case, then dogma mock score.',
        soft_wrap=True,
    )


def _named(title: str | None, identifier: str | None, empty: str = "none") -> str:
    if identifier is None:
        return empty
    return f"{escape(title or identifier)} ({escape(identifier)})"


def _print_report(console: Console, report: MockReport) -> None:
    console.print(
        f"\nScored: {report.total} of {report.maximum} · {escape(report.band.label)}",
        soft_wrap=True,
    )
    console.print(escape(report.band.text), soft_wrap=True)


def _print_debrief(console: Console, view: MockDebrief) -> None:
    report = view.report
    console.print(
        f"[bold]Mock debrief[/] · Total: {report.total} of {report.maximum} · "
        f"{escape(report.band.label)}",
        soft_wrap=True,
    )
    console.print(escape(report.band.text), soft_wrap=True)
    for part in report.parts:
        console.print(f"  {_part_line(part)}", soft_wrap=True)
    console.print(f"Strongest part: {_part_line(report.strongest)}", soft_wrap=True)
    console.print(f"Improve first: {_part_line(report.weakest)}", soft_wrap=True)
    if view.case_debrief is not None:
        console.print("\n[bold]Case debrief[/]")
        console.print(escape(view.case_debrief), soft_wrap=True)


def _part_line(part: PartScore) -> str:
    return f"{escape(part.title)}: {part.score} of {part.maximum} ({round(part.percent * 100)}%)"


app.command("start", epilog="Example: dogma mock start --case sys/churn-prediction-case")(
    friendly_errors(start_command)
)
app.command("reveal", epilog='Example: dogma mock reveal "What does the current process do?"')(
    friendly_errors(reveal_command)
)
app.command("ask", epilog='Example: dogma mock ask "How do the segments differ?"')(
    friendly_errors(ask_command)
)
app.command("score", epilog="Example: dogma mock score --scores-file scores.json --model-score")(
    friendly_errors(score_command)
)
app.command("debrief", epilog="Example: dogma mock debrief")(friendly_errors(debrief_command))
app.command("abandon", epilog="Example: dogma mock abandon")(friendly_errors(abandon_command))
app.command("status", epilog="Example: dogma mock status")(friendly_errors(status_command))
