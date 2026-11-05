"""Local corpus synchronization and citation-bearing search commands."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

import typer
from rich.markup import escape

from dogma.cli.context import err, friendly_errors, out, settings_of
from dogma.corpus.convert import sync_corpus
from dogma.corpus.index import search

app = typer.Typer(
    help="Convert and search local notes, lessons, and papers.",
    no_args_is_help=True,
    add_completion=False,
)


class OcrPolicy(StrEnum):
    """Explicit docmd policy for scanned or garbled PDF pages."""

    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


@app.command("sync", epilog="Example: dogma corpus sync --no-convert")
@friendly_errors
def sync_command(
    ctx: typer.Context,
    convert: Annotated[
        bool, typer.Option("--convert/--no-convert", help="Run vendored docmd before indexing.")
    ] = True,
    ocr: Annotated[
        OcrPolicy, typer.Option(help="OCR policy; auto/force opt into the locked OCR extra.")
    ] = OcrPolicy.OFF,
) -> None:
    """Incrementally convert local documents and refresh the separate corpus index."""
    report = sync_corpus(settings_of(ctx), convert=convert, ocr=ocr.value)
    counts = report.index
    out().print(
        f"Corpus: {counts.added} added, {counts.updated} updated, "
        f"{counts.removed} removed, {counts.unchanged} unchanged; "
        f"{report.conversion.converted} source directories converted."
    )
    for issue in report.issues:
        err().print(escape(f"[{issue.code}] {issue.path}: {issue.message}"))
    if report.issues:
        raise typer.Exit(1)


@app.command("search", epilog='Example: dogma corpus search "Adam optimizer" --limit 5')
@friendly_errors
def search_command(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Words to find as literal terms.")],
    topic: Annotated[list[str] | None, typer.Option("--topic", help="Filter by topic id.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 5,
) -> None:
    """Search the local FTS index and display passages with source/page citations."""
    hits = search(settings_of(ctx), query, topics=tuple(topic or ()), k=limit)
    if not hits:
        out().print("No matches.")
    for hit in hits:
        out().print(escape(hit.citation))
        if hit.chunk.headings:
            out().print(escape(" > ".join(hit.chunk.headings)))
        out().print(escape(hit.chunk.text))
        out().print()
