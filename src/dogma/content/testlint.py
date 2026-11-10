"""Static lint for a problem's hidden tests. Read with ``ast``; nothing is imported or run."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass

RULES: tuple[str, ...] = (
    "HIDDEN_PARAM_IDS",
    "HIDDEN_BARE_RAISES",
    "HIDDEN_GIVEN_SETTINGS",
    "HIDDEN_ASSERT_MESSAGE",
)

_PARAM_IDS_MESSAGE = "parametrize needs ids=: default ids print the hidden inputs in the test name"
_BARE_RAISES_MESSAGE = "use a helper that names the mistake instead of pytest.raises"
_GIVEN_MESSAGE = "@given needs @settings(deadline=None, report_multiple_bugs=False)"
_ASSERT_STATEMENT_MESSAGE = "assert needs a message that names the kind of mistake"
_ASSERT_CALL_NAMES: frozenset[str] = frozenset(
    {
        "assert_allclose",
        "assert_array_equal",
        "assert_array_almost_equal",
        "assert_almost_equal",
        "assert_equal",
        "assert_close",
    }
)
_MESSAGE_KEYWORDS: frozenset[str] = frozenset({"err_msg", "msg"})
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_DTYPE_NAME = re.compile(r"(?i)\b(?:u?int|float|complex|bool|fp|bf)(?:8|16|32|64|128)\b")


@dataclass(frozen=True)
class LintFinding:
    rule: str  # one of RULES
    line: int  # 1-based line in the source
    message: str  # the fixed sentence for the rule


def lint_hidden_tests(source: str) -> list[LintFinding]:
    """Return the findings for one hidden-test file, sorted by line then rule.

    A syntax error gives no findings: the validator reports it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings = [
        *_param_ids(tree),
        *_bare_raises(tree),
        *_given_settings(tree),
        *_assert_messages(tree),
    ]
    return sorted(findings, key=lambda finding: (finding.line, finding.rule))


def id_leaks(names: Iterable[str]) -> list[str]:
    """Return the test names whose ``[...]`` id part still holds a digit after dtype names go."""
    leaks: list[str] = []
    for name in names:
        start, end = name.find("["), name.rfind("]")
        if start == -1 or end < start:
            continue
        inner = _DTYPE_NAME.sub("", name[start + 1 : end])
        if re.search(r"\d", inner):
            leaks.append(name)
    return leaks


def _last_name(call: ast.Call) -> str:
    return ast.unparse(call.func).rsplit(".", 1)[-1]


def _calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _decorator_calls(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> list[ast.Call]:
    return [decorator for decorator in node.decorator_list if isinstance(decorator, ast.Call)]


def _keyword_is_constant(call: ast.Call, name: str, value: object) -> bool:
    return any(
        keyword.arg == name
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is value
        for keyword in call.keywords
    )


def _is_id_param(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and _last_name(node) == "param"
        and any(keyword.arg == "id" for keyword in node.keywords)
    )


def _has_ids(call: ast.Call) -> bool:
    if any(keyword.arg == "ids" for keyword in call.keywords):
        return True
    if len(call.args) < 2:
        return False
    cases = call.args[1]
    return isinstance(cases, ast.List | ast.Tuple) and all(
        _is_id_param(case) for case in cases.elts
    )


def _param_ids(tree: ast.Module) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, _DEFINITIONS):
            continue
        for decorator in _decorator_calls(node):
            if _last_name(decorator) == "parametrize" and not _has_ids(decorator):
                findings.append(
                    LintFinding("HIDDEN_PARAM_IDS", decorator.lineno, _PARAM_IDS_MESSAGE)
                )
    return findings


def _bare_raises(tree: ast.Module) -> list[LintFinding]:
    return [
        LintFinding("HIDDEN_BARE_RAISES", call.lineno, _BARE_RAISES_MESSAGE)
        for call in _calls(tree)
        if ast.unparse(call.func) in ("pytest.raises", "raises")
    ]


def _given_settings(tree: ast.Module) -> list[LintFinding]:
    findings: list[LintFinding] = []
    aliases = _module_call_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue
        decorators = _decorator_calls(node)
        if not any(_last_name(decorator) == "given" for decorator in decorators):
            continue
        if any(_has_clean_settings(decorator) for decorator in decorators):
            continue
        if any(
            isinstance(decorator, ast.Name) and _alias_is_clean(aliases.get(decorator.id))
            for decorator in node.decorator_list
        ):
            continue
        findings.append(LintFinding("HIDDEN_GIVEN_SETTINGS", node.lineno, _GIVEN_MESSAGE))
    return findings


def _module_call_aliases(tree: ast.Module) -> dict[str, list[ast.expr]]:
    """Every value a module-level name is bound to, for names used as bare decorators."""
    aliases: dict[str, list[ast.expr]] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    aliases.setdefault(target.id, []).append(statement.value)
    return aliases


def _alias_is_clean(values: list[ast.expr] | None) -> bool:
    """A bare ``@NAME`` counts only if every module-level binding is a clean settings call."""
    if not values:
        return False
    return all(isinstance(value, ast.Call) and _has_clean_settings(value) for value in values)


def _has_clean_settings(call: ast.Call) -> bool:
    return (
        _last_name(call) == "settings"
        and _keyword_is_constant(call, "deadline", None)
        and _keyword_is_constant(call, "report_multiple_bugs", False)
    )


def _has_message(call: ast.Call) -> bool:
    return any(keyword.arg in _MESSAGE_KEYWORDS for keyword in call.keywords)


def _assert_messages(tree: ast.Module) -> list[LintFinding]:
    findings: list[LintFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and node.msg is None:
            findings.append(
                LintFinding("HIDDEN_ASSERT_MESSAGE", node.lineno, _ASSERT_STATEMENT_MESSAGE)
            )
        elif isinstance(node, ast.Call):
            name = _last_name(node)
            if name in _ASSERT_CALL_NAMES and not _has_message(node):
                findings.append(
                    LintFinding(
                        "HIDDEN_ASSERT_MESSAGE",
                        node.lineno,
                        f"{name} needs err_msg= (msg= for torch)",
                    )
                )
    return findings
