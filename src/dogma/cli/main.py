"""The ``dogma`` command: wires every command to its module (ARCHITECTURE §4.7).

Each command is wrapped by ``friendly_errors`` so that dogma's own errors print as one line.
Every command's help ends with one example.
"""

from __future__ import annotations

import typer

from dogma.cli import (
    answer_cmd,
    backup,
    catalog_cmd,
    close,
    corpus,
    coverage_view,
    db_cmd,
    mock_cmd,
    path_view,
    practice,
    progress,
    resolve_cmd,
    review,
    runs,
    scaffold,
    serve,
    today,
    tutor,
    verify,
    warp_cmd,
)
from dogma.cli.context import friendly_errors
from dogma.core.config import get_settings

ROOT_HELP = "Practise ML, data science and AI-engineering interview problems locally."
ROOT_EPILOG = "Example: dogma open cml/vector-operations/p01-vector-projection"

app = typer.Typer(
    name="dogma",
    help=ROOT_HELP,
    epilog=ROOT_EPILOG,
    no_args_is_help=True,
    add_completion=False,
)
content_app = typer.Typer(
    help="Author and check content: verify, coverage, new.",
    no_args_is_help=True,
    add_completion=False,
)
catalog_app = typer.Typer(
    help="Compile the content catalog.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(
    help="The learner database: upgrade its schema.",
    no_args_is_help=True,
    add_completion=False,
)
course_app = typer.Typer(
    help="Start the course over: reset.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(content_app, name="content")
app.add_typer(catalog_app, name="catalog")
app.add_typer(db_app, name="db")
app.add_typer(corpus.app, name="corpus")
app.add_typer(runs.app, name="runs")
app.add_typer(mock_cmd.app, name="mock")
app.add_typer(course_app, name="course")


@app.callback()
def _root(ctx: typer.Context) -> None:
    """Practise locally: open a problem, run it, submit it, follow the core path."""
    if ctx.obj is None:
        ctx.obj = get_settings()


app.command(
    "open",
    epilog="Example: dogma open cml/vector-operations/p01-vector-projection",
)(friendly_errors(practice.open_command))
app.command(
    "run",
    epilog="Example: dogma run cml/vector-operations/p01-vector-projection",
)(friendly_errors(practice.run_command))
app.command(
    "submit",
    epilog="Example: dogma submit cml/vector-operations/p01-vector-projection",
)(friendly_errors(practice.submit_command))
app.command(
    "path",
    epilog="Example: dogma path --week 1",
)(friendly_errors(path_view.path_command))
app.command(
    "progress",
    epilog="Example: dogma progress --week 1",
)(friendly_errors(progress.progress_command))
app.command(
    "review",
    epilog="Example: dogma review --limit 20",
)(friendly_errors(review.review_command))
app.command(
    "today",
    epilog="Example: dogma today --energy low",
)(friendly_errors(today.today_command))
app.command(
    "close",
    epilog="Example: dogma close --stopped-where 'Practice, example 2' --next-step 'Solve it'",
)(friendly_errors(close.close_command))
app.command(
    "backup",
    epilog="Example: dogma backup",
)(friendly_errors(backup.backup_command))
app.command(
    "hint",
    epilog="Example: dogma hint cml/vector-operations/p01-vector-projection --level 1",
)(friendly_errors(tutor.hint_command))
app.command(
    "debug",
    epilog="Example: dogma debug cml/vector-operations/p01-vector-projection",
)(friendly_errors(tutor.debug_command))
app.command(
    "answer",
    epilog=(
        "Example: dogma answer cml/vector-operations "
        "--scores accuracy=3,structure=3,trade-offs=3,communication=4"
    ),
)(friendly_errors(answer_cmd.answer_command))
app.command(
    "resolve",
    epilog="Example: dogma resolve cml/vector-operations/p01-vector-projection",
)(friendly_errors(resolve_cmd.resolve_command))

content_app.command(
    "verify",
    epilog="Example: dogma content verify cml/vector-operations  (or --all)",
)(friendly_errors(verify.verify_command))
content_app.command(
    "coverage",
    epilog="Example: dogma content coverage --tier A",
)(friendly_errors(coverage_view.coverage_command))
content_app.command(
    "new",
    epilog="Example: dogma content new cml/cosine-similarity",
)(friendly_errors(scaffold.new_command))

catalog_app.command(
    "build",
    epilog="Example: dogma catalog build",
)(friendly_errors(catalog_cmd.build_command))

app.command(
    "serve",
    epilog="Example: dogma serve --port 8765",
)(friendly_errors(serve.serve_command))

db_app.command(
    "upgrade",
    epilog="Example: dogma db upgrade",
)(friendly_errors(db_cmd.upgrade_command))

app.command(
    "warp",
    epilog="Example: dogma warp --to-week 3",
)(friendly_errors(warp_cmd.warp_command))
course_app.command(
    "reset",
    epilog="Example: dogma course reset",
)(friendly_errors(warp_cmd.reset_command))
