"""Versioned prompt templates (ENGINEERING_STANDARDS §10).

Each mode is one file, ``prompts/<mode>.md``: YAML front matter with an integer ``version`` and the
matching ``mode``, then shared rules, then ``## L1`` to ``## L4`` sections for hint levels. A change
to a prompt bumps its version and its golden snapshot in the same commit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dogma.content.markdown import split_front_matter
from dogma.core.errors import DogmaError, ErrorCode
from dogma.tutor.models import CallMode, Message, RenderedPrompt

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
_LEVEL_HEADING = re.compile(r"^## L([1-4])\b.*$", re.MULTILINE)


@dataclass(frozen=True)
class PromptTemplate:
    mode: CallMode
    version: int
    shared: str
    sections: dict[int, str]


def load_template(mode: CallMode, directory: Path = PROMPT_DIR) -> PromptTemplate:
    """Read and validate one mode's prompt file. A bad file is a TUTOR_PROMPT_INVALID error."""
    path = directory / f"{mode.value}.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _invalid(path, "the prompt file is missing or unreadable") from exc
    try:
        front, body = split_front_matter(text)
    except ValueError as exc:
        raise _invalid(path, str(exc)) from exc
    fields = _front_matter(path, front)
    version = fields.get("version")
    if type(version) is not int or version < 1:
        raise _invalid(path, "front matter needs an integer version of at least 1")
    if fields.get("mode") != mode.value:
        raise _invalid(path, f"front matter mode must be {mode.value!r}")
    shared, sections = _split_sections(body)
    return PromptTemplate(mode=mode, version=version, shared=shared.strip(), sections=sections)


def system_text(mode: CallMode, level: int | None, directory: Path = PROMPT_DIR) -> str:
    """The system message: shared rules plus the instructions for this hint level, if any."""
    template = load_template(mode, directory)
    if mode is not CallMode.HINT:
        return template.shared
    if level is None or level not in template.sections:
        raise DogmaError(
            ErrorCode.TUTOR_PROMPT_INVALID, f"hint level {level!r} has no instructions"
        )
    return f"{template.shared}\n\n{template.sections[level]}"


def render_prompt(
    mode: CallMode, level: int | None, user_text: str, directory: Path = PROMPT_DIR
) -> RenderedPrompt:
    """Combine the system text and the assembled context into the messages sent to a model."""
    template = load_template(mode, directory)
    system = system_text(mode, level, directory)
    messages = (Message("system", system), Message("user", user_text))
    digest = hashlib.sha256("\x1e".join(m.content for m in messages).encode("utf-8")).hexdigest()
    return RenderedPrompt(
        mode=mode,
        level=level if mode is CallMode.HINT else None,
        version=template.version,
        sha256=digest,
        messages=messages,
    )


def golden_text(prompt: RenderedPrompt) -> str:
    """The snapshot format: a labelled header, then the system and user messages verbatim."""
    label = prompt.mode.value + (f" L{prompt.level}" if prompt.level is not None else "")
    system, user = prompt.messages[0].content, prompt.messages[1].content
    return (
        f"=== {label} · prompt v{prompt.version} ===\n--- system ---\n{system}\n"
        f"--- user ---\n{user}\n"
    )


def _front_matter(path: Path, front: str | None) -> dict[str, Any]:
    if front is None:
        raise _invalid(path, "missing YAML front matter (a '---' block at the top)")
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError as exc:
        raise _invalid(path, "front matter is not valid YAML") from exc
    if not isinstance(data, dict):
        raise _invalid(path, "front matter must be a mapping")
    return data


def _split_sections(body: str) -> tuple[str, dict[int, str]]:
    matches = list(_LEVEL_HEADING.finditer(body))
    if not matches:
        return body, {}
    sections: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections[int(match.group(1))] = body[match.end() : end].strip()
    return body[: matches[0].start()], sections


def _invalid(path: Path, reason: str) -> DogmaError:
    return DogmaError(ErrorCode.TUTOR_PROMPT_INVALID, f"{path.name}: {reason}")
