"""Static facts about a problem's Python files, read with ``ast``. Nothing is imported or run."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

_NONE: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PythonFacts:
    """What the content gate needs to know about one ``.py`` file.

    ``defined`` holds top-level names (functions, classes, assignments). ``signatures`` maps each
    top-level function to its parameter names in order. ``test_functions`` holds top-level
    ``test_*`` functions in order. ``imported_modules`` holds every module named by an import
    anywhere in the file, and ``submission_names`` the names imported from ``submission``.
    ``string_constants`` holds every string literal. ``raises_not_implemented`` is True when the
    file raises NotImplementedError somewhere (a starter stub).
    """

    syntax_error: str | None
    defined: frozenset[str] = _NONE
    signatures: dict[str, tuple[str, ...]] = field(default_factory=dict)
    test_functions: tuple[str, ...] = ()
    imported_modules: frozenset[str] = _NONE
    submission_names: frozenset[str] = _NONE
    string_constants: frozenset[str] = _NONE
    raises_not_implemented: bool = False


def parse_python_facts(source: str) -> PythonFacts:
    """Parse ``source`` with ``ast``. A syntax error is reported in the result, not raised."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return PythonFacts(syntax_error=f"line {exc.lineno}: {exc.msg}")
    defined, signatures, tests = _top_level(tree)
    modules, submission = _imports(tree)
    return PythonFacts(
        syntax_error=None,
        defined=frozenset(defined),
        signatures=signatures,
        test_functions=tuple(tests),
        imported_modules=frozenset(modules),
        submission_names=frozenset(submission),
        string_constants=frozenset(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ),
        raises_not_implemented=_raises_not_implemented(tree),
    )


def _top_level(tree: ast.Module) -> tuple[set[str], dict[str, tuple[str, ...]], list[str]]:
    defined: set[str] = set()
    signatures: dict[str, tuple[str, ...]] = {}
    tests: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(node.name)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                signatures[node.name] = _parameter_names(node.args)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                tests.append(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
    return defined, signatures, tests


def _parameter_names(args: ast.arguments) -> tuple[str, ...]:
    ordered = [*args.posonlyargs, *args.args]
    if args.vararg is not None:
        ordered.append(args.vararg)
    ordered.extend(args.kwonlyargs)
    if args.kwarg is not None:
        ordered.append(args.kwarg)
    return tuple(arg.arg for arg in ordered)


def _imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    modules: set[str] = set()
    submission: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            if node.module == "submission":
                submission.update(alias.name for alias in node.names)
    return modules, submission


def _raises_not_implemented(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, ast.Name) and target.id == "NotImplementedError":
                return True
    return False
