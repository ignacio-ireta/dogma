"""The ceiling guard for the nudge (I-22): formulas that only a later hint level gives are held
back.

A formula is a span of LaTeX math (``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]``) or of
inline code. A span counts only when it has at least one operator and at least three distinct
tokens.

Tokens are identifiers, numbers and operator symbols. LaTeX macros become symbols: ``\\cdot`` is
the dot, ``\\dfrac`` and ``\\frac`` are a slash. Other macros, braces, parentheses, spacing and
punctuation are dropped. A formula is the *set* of its tokens, so order and repeats do not count.

For each formula F in a nudge sentence:

- F is allowed when it is a subset of a formula the learner may already see: the problem
  statement, or a reviewed hint at or below the level asked for.
- Otherwise F is held back when it equals, contains or is contained in a formula that only a
  later level gives. The sentence is removed, and the caller adds a note that a formula was left
  out.

Limits, on purpose: this is token matching, not algebra. It misses a formula written with other
symbols or names. It over-strips a fragment of a later formula, such as the dot product of two
vectors taken out of a larger expression, unless the problem statement already shows that
fragment. Text outside formula spans is never judged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_SPAN = re.compile(
    r"\$\$(?P<display>.+?)\$\$"
    r"|\\\((?P<paren>.+?)\\\)"
    r"|\\\[(?P<bracket>.+?)\\\]"
    r"|`(?P<code>[^`\n]+)`"
    r"|\$(?P<inline>[^$\n]+?)\$",
    re.DOTALL,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MACRO = re.compile(r"\\([A-Za-z]+)")
_ESCAPE = re.compile(r"\\.", re.DOTALL)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[=+\-*/^@·\u00d7÷√∑∏<>≤≥≠]")
_SYMBOL_MACROS = {
    "cdot": "·",
    "times": "\u00d7",
    "div": "÷",
    "frac": "/",
    "dfrac": "/",
    "tfrac": "/",
    "sum": "∑",
    "prod": "∏",
    "sqrt": "√",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ne": "≠",
    "neq": "≠",
}
_TEXT_SYMBOLS = (("!=", "≠"), ("<=", "≤"), (">=", "≥"), ("==", "="))
OPERATORS = frozenset("=+-*/^@·\u00d7÷√∑∏<>≤≥≠")
MIN_TOKENS = 3
FORMULA_NOTE = "(A formula was left out: it belongs to a later hint level.)"

Form = frozenset[str]


def formulas(text: str) -> tuple[Form, ...]:
    """Every formula in the text, as token sets, in reading order."""
    forms: list[Form] = []
    for match in _SPAN.finditer(text):
        form = token_set(_span_body(match))
        if form is not None:
            forms.append(form)
    return tuple(forms)


def token_set(span: str) -> Form | None:
    """The tokens of one span, or None when it is not a formula (no operator, or too few tokens)."""
    text = _ESCAPE.sub(" ", _MACRO.sub(_macro, span))
    for old, new in _TEXT_SYMBOLS:
        text = text.replace(old, new)
    tokens = frozenset(_TOKEN.findall(text))
    if len(tokens) < MIN_TOKENS or not tokens & OPERATORS:
        return None
    return tokens


def guard_ceiling(
    text: str, *, shown: Iterable[str], above: Iterable[str], problem: str
) -> tuple[str, bool]:
    """Remove the sentences whose formulas only a later hint level gives. Returns (body, changed).

    ``shown`` are the reviewed hint texts at or below the level asked for, ``above`` the texts of
    later levels, and ``problem`` the problem statement. Only ``above`` decides what is held back.
    When something was removed, the caller adds ``FORMULA_NOTE`` to the body. The body may be empty.
    """
    permitted = [form for body in (*shown, problem) for form in formulas(body)]
    leaked = [form for body in above for form in formulas(body) if not _within(form, permitted)]
    if not leaked:
        return text, False
    changed = False
    lines: list[str] = []
    for line in text.split("\n"):
        kept: list[str] = []
        for sentence in _SENTENCE_END.split(line):
            if any(_leaks(form, leaked, permitted) for form in formulas(sentence)):
                changed = True
            else:
                kept.append(sentence)
        lines.append(" ".join(kept))
    if not changed:
        return text, False
    return "\n".join(lines).strip(), True


def _span_body(match: re.Match[str]) -> str:
    return next(body for body in match.groups() if body is not None)


def _macro(match: re.Match[str]) -> str:
    return " " + _SYMBOL_MACROS.get(match.group(1), " ") + " "


def _within(form: Form, forms: Iterable[Form]) -> bool:
    return any(form <= other for other in forms)


def _leaks(form: Form, leaked: list[Form], permitted: list[Form]) -> bool:
    if _within(form, permitted):
        return False
    return any(form <= other or other <= form for other in leaked)
