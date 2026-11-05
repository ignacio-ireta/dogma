"""Bounded, deterministic prompt assembly (ARCHITECTURE §4.5: under about 12K tokens).

Blocks come in priority order: problem, report, learner file, lesson sections, corpus passages.
Each block has a cap, and the whole prompt has a budget. A block that no longer fits is omitted and
named in the notes. Citable blocks get tags S1, S2, ... in the order they are kept. Nothing here
reads a file or the database; ``evidence.py`` does that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dogma.corpus.models import SearchHit
from dogma.grading.models import GradeReport
from dogma.tutor.evidence import Evidence, History
from dogma.tutor.hints import REVIEWED_LEVELS
from dogma.tutor.models import CallMode, RenderedPrompt, ReviewedHint, Source, SourceKind
from dogma.tutor.prompts import PROMPT_DIR, render_prompt, system_text

CHARS_PER_TOKEN = 3  # a conservative estimate for mixed prose, code and maths
TOKEN_BUDGET = 12_000
MAX_PROMPT_CHARS = TOKEN_BUDGET * CHARS_PER_TOKEN
CORPUS_SLOTS = 4  # corpus passages in one prompt, at most one per source document
_MIN_BLOCK_CHARS = 200
_SUMMARY_RESERVE = 1_000
_TRUNCATED = "\n[truncated]"
_CAP_PROBLEM = 9_000
_CAP_REPORT = 3_000
_CAP_WORKSPACE = 6_000
_CAP_HINT = 1_500
_CAP_SECTION = 3_000
_CAP_HIT = 1_500
_PUBLIC_MESSAGE_CHARS = 300


@dataclass(frozen=True)
class AssembledPrompt:
    prompt: RenderedPrompt
    sources: tuple[Source, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class _Block:
    name: str  # short name for the notes when the block is omitted
    label: str  # may contain "{tag}", filled in once the block is kept
    body: str
    cap: int
    kind: SourceKind | None = None
    citation: str | None = None


def assemble_prompt(
    evidence: Evidence,
    *,
    mode: CallMode,
    level: int | None = None,
    directory: Path = PROMPT_DIR,
) -> AssembledPrompt:
    """Render the messages for one call, within the budget, with its citable sources."""
    ceiling = _ceiling(evidence, mode, level)
    task = _task(mode, level, grounded=bool(ceiling))
    room = MAX_PROMPT_CHARS - len(system_text(mode, level, directory))
    room -= len(task) + 2 + _SUMMARY_RESERVE + 2
    sections: list[str] = []
    sources: list[Source] = []
    omitted: list[str] = []
    for block in _blocks(evidence, ceiling):
        tag = f"S{len(sources) + 1}" if block.citation is not None else ""
        head = f"### {block.label.replace('{tag}', tag)}"
        allowance = min(block.cap, room - len(head) - 3)
        if allowance < _MIN_BLOCK_CHARS:
            omitted.append(block.name)
            continue
        body = _truncate(block.body.rstrip(), allowance)
        sections.append(f"{head}\n{body}")
        room -= len(head) + len(body) + 3
        if block.citation is not None and block.kind is not None:
            sources.append(Source(tag=tag, citation=block.citation, kind=block.kind))
    user = "\n\n".join([task, *sections, _summary(sources)])
    notes = evidence.notes
    if omitted:
        notes = (*notes, f"omitted for length: {', '.join(omitted)}")
    prompt = render_prompt(mode, level, user, directory)
    return AssembledPrompt(prompt=prompt, sources=tuple(sources), notes=notes)


def render_report(report: GradeReport, *, submitted_at: str | None) -> str:
    """A plain-text view of a report: status, counts and each test's outcome. No hidden values."""
    lines = [f"status: {report.status.value}", f"submitted: {submitted_at or 'unknown'}"]
    if report.tests:
        public_passed, public_total = report.counts("public")
        hidden_passed, hidden_total = report.counts("hidden")
        lines.append(
            f"public {public_passed}/{public_total} passed; "
            f"hidden {hidden_passed}/{hidden_total} passed"
        )
    for test in report.tests:
        line = f"- {test.name} ({test.file}): {test.outcome}"
        message = _one_line(test.message)
        if message:
            line += f" -- {message}"
        lines.append(line)
    for finding in report.static:
        where = f" line {finding.line}" if finding.line is not None else ""
        lines.append(f"- static check {finding.rule}{where}: {_one_line(finding.message)}")
    if report.message:
        lines.append(f"message: {_one_line(report.message)}")
    return "\n".join(lines)


def select_corpus(hits: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
    """At most one passage per source document, the best-ranked one, and at most CORPUS_SLOTS.

    ``hits`` are in search order, best first, so the first passage of a document is its best one.
    """
    chosen: list[SearchHit] = []
    documents: set[str] = set()
    for hit in hits:
        if hit.chunk.source_id in documents:
            continue
        documents.add(hit.chunk.source_id)
        chosen.append(hit)
        if len(chosen) == CORPUS_SLOTS:
            break
    return tuple(chosen)


def _ceiling(evidence: Evidence, mode: CallMode, level: int | None) -> tuple[ReviewedHint, ...]:
    """The reviewed hints a hint call may restate: levels up to the one asked for, never more."""
    if mode is not CallMode.HINT or level not in REVIEWED_LEVELS:
        return ()
    return evidence.reviewed.up_to(level)


def _blocks(evidence: Evidence, ceiling: Sequence[ReviewedHint]) -> tuple[_Block, ...]:
    """The context blocks in priority order. Both modes take the same context."""
    language = "sql" if evidence.language == "sql" else "python"
    blocks: list[_Block] = [
        _Block(
            name="problem",
            label=f"PROBLEM ({evidence.problem_id}, {evidence.title}, {evidence.language})",
            body=evidence.problem_text or "(no problem text)",
            cap=_CAP_PROBLEM,
        ),
        _Block(
            name="report",
            label="REPORT (latest recorded attempt)",
            body=_report_body(evidence.history),
            cap=_CAP_REPORT,
        ),
        _Block(
            name="learner file",
            label=f"LEARNER FILE ({evidence.workspace_name})",
            body=_code(evidence.workspace_text, language)
            if evidence.workspace_text is not None
            else "(no workspace file yet)",
            cap=_CAP_WORKSPACE,
        ),
    ]
    for hint in ceiling:
        blocks.append(
            _Block(
                name="reviewed hint",
                label=f"REVIEWED HINT {{tag}}: Hint {hint.level} of 3, {hint.label}",
                body=hint.text,
                cap=_CAP_HINT,
                kind="reviewed",
                citation=f"hints · {evidence.problem_id} · Hint {hint.level}: {hint.label}",
            )
        )
    for section in evidence.lesson_sections:
        citation = f"lesson · {evidence.topic_id} · {section.heading}"
        blocks.append(
            _Block(
                name="lesson section",
                label=f"SOURCE {{tag}}: {citation}",
                body=section.text,
                cap=_CAP_SECTION,
                kind="lesson",
                citation=citation,
            )
        )
    for hit in select_corpus(evidence.corpus_hits):
        chunk = hit.chunk
        plain = chunk.source_id + (f" p.{chunk.page}" if chunk.page is not None else "")
        headings = f"Headings: {' > '.join(chunk.headings)}\n" if chunk.headings else ""
        blocks.append(
            _Block(
                name="corpus passage",
                label=f"SOURCE {{tag}}: corpus · {plain}",
                body=headings + chunk.text,
                cap=_CAP_HIT,
                kind="corpus",
                citation=f"corpus · {plain}",
            )
        )
    return tuple(blocks)


def _task(mode: CallMode, level: int | None, *, grounded: bool) -> str:
    if mode is CallMode.HINT and grounded:
        return (
            f"TASK: hint at level L{level}. The REVIEWED HINT blocks are the ceiling for this "
            "call: restate them, relate them to the LEARNER FILE and the REPORT, and go no "
            "further. Cite SOURCES."
        )
    if mode is CallMode.HINT:
        return (
            f"TASK: hint at level L{level}. Follow the L{level} section of the instructions "
            "and cite SOURCES."
        )
    return "TASK: debug the latest recorded attempt. Use the REPORT and the LEARNER FILE."


def _summary(sources: list[Source]) -> str:
    if not sources:
        return "SOURCES: none available; write unsourced for every claim."
    listed = "; ".join(f"{source.tag} = {source.citation}" for source in sources)
    return _truncate(f"SOURCES: {listed}", _SUMMARY_RESERVE)


def _report_body(history: History) -> str:
    if history.latest is None:
        return "no recorded attempt for this problem yet"
    return render_report(history.latest, submitted_at=history.latest_submitted_at)


def _code(text: str, language: str) -> str:
    fence = "```" if "```" not in text else "~~~~"
    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


def _one_line(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= _PUBLIC_MESSAGE_CHARS:
        return flat
    return flat[: _PUBLIC_MESSAGE_CHARS - 3] + "..."


def _truncate(text: str, limit: int) -> str:
    """Cut to at most ``limit`` characters, at a line break when one is near the end."""
    if len(text) <= limit:
        return text
    if limit <= len(_TRUNCATED):
        return text[: max(limit, 0)]
    cut = text[: limit - len(_TRUNCATED)]
    newline = cut.rfind("\n")
    if newline > len(cut) // 2:
        cut = cut[:newline]
    return cut + _TRUNCATED
