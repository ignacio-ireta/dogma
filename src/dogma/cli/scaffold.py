"""``dogma content new``: scaffold a topic package from templates/ (CONTENT_SPEC §1).

The templates are the example topic with placeholder text. Every file is rendered in memory first,
then written, so a failure leaves no half-made package. The scaffold loads (its schemas are valid)
but does not verify until its TODOs are filled, which is intended.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from dogma.cli.context import CliError, out, settings_of
from dogma.content.loader import load_syllabus
from dogma.core.errors import ContentError, ErrorCode
from dogma.core.fsio import atomic_write

TEMPLATE_DIR = Path("templates") / "topic"
CASE_TEMPLATE_DIR = Path("templates") / "case"
TEMPLATE_SUFFIX = ".tmpl"


def new_command(
    ctx: typer.Context,
    topic_id: Annotated[
        str, typer.Argument(help="A topic id from the syllabus, e.g. cml/cosine-similarity.")
    ],
) -> None:
    """Create content/<topic id>/ from templates/, with TODO placeholders to fill in."""
    settings = settings_of(ctx)
    syllabus = load_syllabus(settings.syllabus_path)
    entry = syllabus.topics_by_id().get(topic_id)
    if entry is None:
        raise CliError(
            f"{topic_id} is not in curriculum/syllabus.yaml; use a topic id from the syllabus"
        )
    target = settings.content_dir / topic_id
    if target.exists():
        raise CliError(f"{target} already exists; not overwritten")
    is_case = entry.kind == "case"
    files = build_scaffold(
        settings.repo_root / (CASE_TEMPLATE_DIR if is_case else TEMPLATE_DIR),
        topic_id=topic_id,
        title=entry.title,
        tier=entry.tier,
    )
    for rel, text in files.items():
        atomic_write(target / rel, text)
    console = out()
    console.print(f"Scaffolded {escape(str(target))}:", soft_wrap=True)
    for rel in files:
        console.print(f"  {escape(str(target / rel))}", soft_wrap=True)
    if is_case:
        console.print("Next: fill in every TODO (a case has no problems/ to rename), then run:")
    else:
        console.print(
            "Next: rename problems/p01-todo to problems/p01-<slug> (and its meta.yaml id), fill"
            " in every TODO, then run:"
        )
    console.print(f"  dogma content verify {topic_id}")


def build_scaffold(template_root: Path, *, topic_id: str, title: str, tier: str) -> dict[Path, str]:
    """Render every ``*.tmpl`` under ``template_root`` into {relative path: text}, in memory."""
    if not template_root.is_dir():
        raise ContentError(
            ErrorCode.CONTENT_NOT_FOUND,
            f"{template_root} not found; the scaffold needs the templates/ directory",
        )
    values = {
        "TOPIC_ID": topic_id,
        "TITLE": title,
        "TITLE_YAML": json.dumps(title, ensure_ascii=False),
        "TIER": tier,
    }
    files: dict[Path, str] = {}
    for source in sorted(template_root.rglob(f"*{TEMPLATE_SUFFIX}")):
        text = source.read_text(encoding="utf-8")
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", value)
        rel = str(source.relative_to(template_root))
        files[Path(rel[: -len(TEMPLATE_SUFFIX)])] = text
    return files
