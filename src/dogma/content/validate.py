"""Cross-file content rules (CONTENT_SPEC section by section). Every rule returns Issues.

Issue codes are short stable strings. Where a stable ErrorCode already names the family, that
code is used instead: CONTENT_SCHEMA_INVALID (a file failed to parse or validate, reported when
validate_all or validate_repo can load a package and loading it failed), CONTENT_ID_MISMATCH
(a directory, meta.id or topic.yaml id disagrees with its path or syllabus id) and
CONTENT_PREREQ_CYCLE (the requires graph has a cycle).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dogma.content.loader import (
    CasePackage,
    ProblemPackage,
    TopicPackage,
    discover_topic_dirs,
    load_core_path,
    load_syllabus,
    load_topic,
)
from dogma.content.pyfacts import PythonFacts
from dogma.content.schemas import PROBLEM_DIR_PATTERN, CorePath, Syllabus
from dogma.core.config import Settings
from dogma.core.errors import ContentError, ErrorCode

LESSON_SECTIONS = (
    "The system",
    "Worked example",
    "Why it works",
    "Signal vs. noise",
    "Where it breaks",
    "Interview angle",
    "Connections",
)
PROBLEM_SECTIONS = ("Contract", "Examples", "Constraints", "Done looks like")
COMMON_MISTAKES = "Common mistakes"
COMMON_PROBLEM_FILES = ("problem.md", "meta.yaml", "test_public.py", "test_hidden.py", "hints.md")
LANGUAGE_PROBLEM_FILES = {
    "python": ("starter.py", "solution.py"),
    "sql": ("starter.sql", "solution.sql"),
}
PYTHON_ENTRY_FILES = ("starter.py", "solution.py")
# Tests may import the stdlib (any module in sys.stdlib_module_names) and these top-level packages
# (CONTENT_SPEC section 6.4). torch is added only for py-torch problems.
TEST_IMPORT_ALLOWED = frozenset(
    {
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "statsmodels",
        "duckdb",
        "pytest",
        "hypothesis",
        "submission",
        "dogma_data",
    }
)
TEST_IMPORT_TORCH_RUNTIME = "py-torch"
# Allowed top-level packages whose listed submodules are still forbidden, with the reason.
TEST_IMPORT_FORBIDDEN = {
    "statsmodels.api": "it takes about 3 s to import; use statsmodels.datasets.<name> or a module",
    "datasets": "dataset downloads belong to the sync script; tests read the vendored files",
    "huggingface_hub": "downloads belong to the sync script; tests read the vendored files",
}
CARD_MIN, CARD_MAX = 4, 8
QUESTION_MIN, QUESTION_MAX = 2, 4
PROBLEMS_MIN, PROBLEMS_MAX = 1, 3
PUBLIC_TESTS_MIN, HIDDEN_TESTS_MIN = 3, 5
# Case packages (CONTENT_SPEC section 9). The debrief follows the RASS parts of the mock rubric.
CASE_FILES = ("brief.md", "reveals.yaml", "twist.md", "debrief.md")
CASE_ONLY_NAMES = (*CASE_FILES, "exhibits")
DEBRIEF_SECTIONS = ("Reframe", "Ask", "Structure", "Synthesize")
REVEAL_MIN, REVEAL_MAX = 2, 8
EXHIBIT_SUFFIXES = (".png", ".csv")

_CARD_NUMBER = re.compile(r"[1-9][0-9]*")
_QUESTION = re.compile(r"Q(\d+)\. \S")
_PROBLEM_DIR = re.compile(PROBLEM_DIR_PATTERN)
_TODO = re.compile(r"\bTODO\b")
_TEXT_SUFFIXES = frozenset({".md", ".yaml", ".py", ".sql"})
# ``library:`` names a note under sources/library/; ``source:`` any other file under sources/.
_SOURCE_ROOTS = {"library": ("sources", "library"), "source": ("sources",)}


@dataclass(frozen=True)
class Issue:
    """One rule violation. ``path`` is the file or directory that needs fixing."""

    code: str
    path: str
    message: str


def validate_topic(pkg: TopicPackage, syllabus: Syllabus) -> list[Issue]:
    """Check one topic package against the syllabus (CONTENT_SPEC sections 2-6 and 9).

    A ``kind: case`` package has no problems and an optional interview; a topic has no case files.
    """
    shared = [
        *_check_syllabus_entry(pkg, syllabus),
        *_check_dir(pkg),
        *_check_requires(pkg, syllabus),
        *_check_sources(pkg),
        *_check_lesson(pkg),
        *_check_cards(pkg),
    ]
    if pkg.meta.kind == "case":
        interview = _check_interview(pkg) if pkg.interview_present else []
        return [*shared, *interview, *_check_case(pkg)]
    return [
        *shared,
        *_check_interview(pkg),
        *_check_problems(pkg),
        *_check_not_case(pkg),
        *_check_placeholders(pkg),
    ]


def validate_path(core_path: CorePath, syllabus: Syllabus) -> list[Issue]:
    """Check a learning path: known topics, no duplicates, and exactly the tier-A topics."""
    known = syllabus.topics_by_id()
    label = f"path:{core_path.id}"
    listed = [topic for week in core_path.weeks for topic in week.topics]
    listed_set = set(listed)
    tier_a = {topic.id for topic in known.values() if topic.tier == "A"}
    issues: list[Issue] = []
    for topic_id in listed:
        if topic_id not in known:
            issues.append(
                Issue("PATH_TOPIC_NOT_IN_SYLLABUS", label, f"{topic_id} is not in the syllabus")
            )
    for topic_id, count in sorted(Counter(listed).items()):
        if count > 1:
            issues.append(
                Issue("PATH_DUPLICATE_TOPIC", label, f"{topic_id} is listed {count} times")
            )
    for topic_id in sorted(tier_a - listed_set):
        issues.append(
            Issue("PATH_MISSING_TIER_A", label, f"tier-A topic {topic_id} is not on the path")
        )
    for topic_id in sorted((listed_set - tier_a) & known.keys()):
        issues.append(
            Issue("PATH_NOT_TIER_A", label, f"{topic_id} is on the path but is not tier A")
        )
    return issues


def validate_prereq_graph(packages: Sequence[TopicPackage]) -> list[Issue]:
    """Check that topic ids are unique and that the requires graph over loaded topics is acyclic.

    Edges to topics without a loaded package are ignored: such topics have no outgoing edges,
    so they cannot close a cycle.
    """
    counts = Counter(pkg.meta.id for pkg in packages)
    issues = [
        Issue(
            "TOPIC_DUPLICATE",
            ", ".join(str(pkg.dir) for pkg in packages if pkg.meta.id == topic_id),
            f"{topic_id} is defined by {count} packages",
        )
        for topic_id, count in sorted(counts.items())
        if count > 1
    ]
    graph = {pkg.meta.id: {dep for dep in pkg.meta.requires if dep in counts} for pkg in packages}
    cycle = _find_cycle(graph)
    if cycle is not None:
        dirs = {pkg.meta.id: str(pkg.dir) for pkg in packages}
        issues.append(
            Issue(
                ErrorCode.CONTENT_PREREQ_CYCLE.value,
                dirs[cycle[0]],
                "prerequisite cycle: " + " -> ".join(cycle),
            )
        )
    return issues


def validate_all(content_dir: Path, syllabus: Syllabus) -> list[Issue]:
    """Load and validate every topic package under ``content_dir``, then the prereq graph."""
    issues: list[Issue] = []
    packages: list[TopicPackage] = []
    for topic_dir in discover_topic_dirs(content_dir):
        try:
            pkg = load_topic(topic_dir)
        except ContentError as exc:
            issues.append(_issue_from(exc, topic_dir))
            continue
        packages.append(pkg)
        issues.extend(validate_topic(pkg, syllabus))
    issues.extend(validate_prereq_graph(packages))
    return issues


def validate_repo(settings: Settings) -> list[Issue]:
    """Validate the syllabus, every learning path and all of ``content/``.

    Raises ContentError when the syllabus or the set of learning paths cannot be loaded at all,
    because nothing else can be checked without them.
    """
    syllabus = load_syllabus(settings.syllabus_path)
    paths_dir = settings.curriculum_dir / "paths"
    path_files = sorted(paths_dir.glob("*.yaml"))
    if not path_files:
        raise ContentError(ErrorCode.CONTENT_NOT_FOUND, f"{paths_dir}: no learning path files")
    issues: list[Issue] = []
    for path_file in path_files:
        try:
            core = load_core_path(path_file)
        except ContentError as exc:
            issues.append(_issue_from(exc, path_file))
            continue
        issues.extend(validate_path(core, syllabus))
    issues.extend(validate_all(settings.content_dir, syllabus))
    return issues


def _issue_from(exc: ContentError, path: Path) -> Issue:
    return Issue(exc.code.value, str(path), exc.message)


def _file(pkg: TopicPackage, name: str) -> str:
    return str(pkg.dir / name)


def _check_syllabus_entry(pkg: TopicPackage, syllabus: Syllabus) -> list[Issue]:
    path = _file(pkg, "topic.yaml")
    entry = syllabus.topics_by_id().get(pkg.meta.id)
    if entry is None:
        return [Issue("TOPIC_NOT_IN_SYLLABUS", path, f"{pkg.meta.id} is not in the syllabus")]
    issues: list[Issue] = []
    if pkg.meta.title != entry.title:
        issues.append(
            Issue(
                "TOPIC_TITLE_MISMATCH",
                path,
                f"title {pkg.meta.title!r} does not match syllabus {entry.title!r}",
            )
        )
    if pkg.meta.tier != entry.tier:
        issues.append(
            Issue(
                "TOPIC_TIER_MISMATCH",
                path,
                f"tier {pkg.meta.tier} does not match syllabus tier {entry.tier}",
            )
        )
    if pkg.meta.kind != entry.kind:
        issues.append(
            Issue(
                "TOPIC_KIND_MISMATCH",
                path,
                f"kind {pkg.meta.kind} does not match syllabus kind {entry.kind}",
            )
        )
    return issues


def _check_dir(pkg: TopicPackage) -> list[Issue]:
    expected = tuple(pkg.meta.id.split("/"))
    if tuple(pkg.dir.parts[-2:]) != expected:
        return [
            Issue(
                ErrorCode.CONTENT_ID_MISMATCH.value,
                str(pkg.dir),
                f"topic directory must end with content/{pkg.meta.id}, found {pkg.dir}",
            )
        ]
    return []


def _check_requires(pkg: TopicPackage, syllabus: Syllabus) -> list[Issue]:
    known = syllabus.topics_by_id()
    return [
        Issue(
            "TOPIC_REQUIRES_UNKNOWN",
            _file(pkg, "topic.yaml"),
            f"requires {dep}, which is not in the syllabus",
        )
        for dep in pkg.meta.requires
        if dep not in known
    ]


def _check_sources(pkg: TopicPackage) -> list[Issue]:
    """``library:`` and ``source:`` citations must name a file that exists in this repository."""
    repo = pkg.dir.parents[2]  # content/<track>/<slug>
    issues: list[Issue] = []
    for item in pkg.meta.sources:
        prefix, _, rest = item.partition(": ")
        if prefix not in _SOURCE_ROOTS:
            continue
        target = repo.joinpath(*_SOURCE_ROOTS[prefix], rest.strip())
        if not target.is_file():
            issues.append(
                Issue("SOURCE_NOT_FOUND", _file(pkg, "topic.yaml"), f"no file for source {item!r}")
            )
    return issues


def _check_placeholders(pkg: TopicPackage) -> list[Issue]:
    """Nothing from the scaffold may survive: no ``TODO`` text and no ``pNN-todo`` directory."""
    issues = [
        Issue("TOPIC_TODO", str(path), "TODO placeholder left from the template")
        for path in _authored_files(pkg.dir)
        if _TODO.search(path.read_text(encoding="utf-8", errors="replace"))
    ]
    issues.extend(
        Issue("TOPIC_TODO", str(problem.dir), "rename the scaffolded pNN-todo directory")
        for problem in pkg.problems
        if problem.dir.name.endswith("-todo")
    )
    return issues


def _authored_files(topic_dir: Path) -> list[Path]:
    """Text files the author writes; fixtures and caches are data, not prose."""
    return sorted(
        path
        for path in topic_dir.rglob("*")
        if path.suffix in _TEXT_SUFFIXES
        and path.is_file()
        and not path.is_symlink()
        and not {"fixtures", "__pycache__"} & set(path.relative_to(topic_dir).parts)
    )


def _check_lesson(pkg: TopicPackage) -> list[Issue]:
    path = _file(pkg, "lesson.md")
    if pkg.lesson is None:
        return [Issue("LESSON_MISSING", path, "lesson.md is missing")]
    if pkg.lesson.headings != LESSON_SECTIONS:
        return [
            Issue(
                "LESSON_SECTIONS",
                path,
                f"H2 sections must be exactly {list(LESSON_SECTIONS)} in order; "
                f"found {list(pkg.lesson.headings)}",
            )
        ]
    return []


def _check_cards(pkg: TopicPackage) -> list[Issue]:
    path = _file(pkg, "cards.yaml")
    if pkg.cards is None:
        return [Issue("CARDS_MISSING", path, "cards.yaml is missing")]
    issues: list[Issue] = []
    if not CARD_MIN <= len(pkg.cards) <= CARD_MAX:
        issues.append(
            Issue(
                "CARD_COUNT",
                path,
                f"a topic needs {CARD_MIN}-{CARD_MAX} cards; found {len(pkg.cards)}",
            )
        )
    prefix = f"{pkg.meta.id}#"
    seen: set[str] = set()
    for card in pkg.cards:
        if not _is_card_id(card.id, prefix):
            issues.append(
                Issue("CARD_ID_FORMAT", path, f"card id {card.id!r} must be {prefix}<n>, n >= 1")
            )
        if card.id in seen:
            issues.append(Issue("CARD_ID_DUPLICATE", path, f"card id {card.id!r} appears twice"))
        seen.add(card.id)
    return issues


def _is_card_id(card_id: str, prefix: str) -> bool:
    return card_id.startswith(prefix) and _CARD_NUMBER.fullmatch(card_id[len(prefix) :]) is not None


def _check_interview(pkg: TopicPackage) -> list[Issue]:
    path = _file(pkg, "interview.md")
    if pkg.interview is None:
        return [Issue("INTERVIEW_MISSING", path, "interview.md is missing")]
    questions = [heading for heading in pkg.interview.headings if heading != COMMON_MISTAKES]
    numbers = [_question_number(heading) for heading in questions]
    in_order = numbers == list(range(1, len(questions) + 1))
    if not in_order or not QUESTION_MIN <= len(questions) <= QUESTION_MAX:
        return [
            Issue(
                "INTERVIEW_QUESTIONS",
                path,
                f"need {QUESTION_MIN}-{QUESTION_MAX} questions headed 'Q1. ...', 'Q2. ...' in "
                f"order; found {len(questions)}",
            )
        ]
    return []


def _question_number(heading: str) -> int | None:
    match = _QUESTION.match(heading)
    return int(match.group(1)) if match else None


def _check_problems(pkg: TopicPackage) -> list[Issue]:
    path = str(pkg.dir / "problems")
    count = len(pkg.problems)
    issues: list[Issue] = []
    if not PROBLEMS_MIN <= count <= PROBLEMS_MAX:
        issues.append(
            Issue(
                "TOPIC_PROBLEM_COUNT",
                path,
                f"a topic has {PROBLEMS_MIN}-{PROBLEMS_MAX} problems; found {count}",
            )
        )
    if pkg.problems:
        issues.extend(_check_core(pkg))
    for problem in pkg.problems:
        issues.extend(_check_problem(pkg, problem))
    return issues


def _check_core(pkg: TopicPackage) -> list[Issue]:
    core = [p for p in pkg.problems if p.meta is not None and p.meta.core]
    if len(core) != 1:
        return [
            Issue(
                "PROBLEM_CORE_COUNT",
                str(pkg.dir / "problems"),
                f"exactly one problem must set core: true; found {len(core)}",
            )
        ]
    if not core[0].dir.name.startswith("p01-"):
        return [Issue("PROBLEM_CORE_NOT_P01", str(core[0].dir), "the core problem must be p01")]
    return []


def _check_problem(topic: TopicPackage, problem: ProblemPackage) -> list[Issue]:
    issues: list[Issue] = []
    if not _PROBLEM_DIR.fullmatch(problem.dir.name):
        issues.append(
            Issue(
                "PROBLEM_DIR_NAME",
                str(problem.dir),
                f"problem directory {problem.dir.name!r} must be pNN-slug (a-z, 0-9, -)",
            )
        )
    issues.extend(_check_problem_files(problem))
    issues.extend(_check_problem_meta(topic, problem))
    issues.extend(_check_problem_md(problem))
    issues.extend(_check_hints(problem))
    issues.extend(_check_python(problem))
    return issues


def _check_problem_files(problem: ProblemPackage) -> list[Issue]:
    language_files = (
        LANGUAGE_PROBLEM_FILES[problem.meta.language] if problem.meta is not None else ()
    )
    required = (*COMMON_PROBLEM_FILES, *language_files)
    return [
        Issue("PROBLEM_FILE_MISSING", str(problem.dir / name), f"required file {name} is missing")
        for name in required
        if name not in problem.files
    ]


def _check_problem_meta(topic: TopicPackage, problem: ProblemPackage) -> list[Issue]:
    if problem.meta is None:
        return []
    expected = f"{topic.meta.id}/{problem.dir.name}"
    if problem.meta.id != expected:
        return [
            Issue(
                ErrorCode.CONTENT_ID_MISMATCH.value,
                str(problem.dir / "meta.yaml"),
                f"meta.yaml id {problem.meta.id!r} must be {expected!r}",
            )
        ]
    return []


def _check_problem_md(problem: ProblemPackage) -> list[Issue]:
    if problem.problem is None:
        return []
    path = str(problem.dir / "problem.md")
    outline = problem.problem
    issues: list[Issue] = []
    if outline.title is None:
        issues.append(Issue("PROBLEM_MD_SECTIONS", path, "problem.md needs a '# <Title>' line"))
    if outline.headings != PROBLEM_SECTIONS:
        issues.append(
            Issue(
                "PROBLEM_MD_SECTIONS",
                path,
                f"H2 sections must be exactly {list(PROBLEM_SECTIONS)} in order; "
                f"found {list(outline.headings)}",
            )
        )
    examples = max((s.examples for s in outline.sections if s.heading == "Examples"), default=0)
    if examples < 2:
        issues.append(
            Issue("PROBLEM_EXAMPLES", path, f"Examples needs at least 2; found {examples}")
        )
    if problem.meta is not None:
        constraints = "\n".join(s.text for s in outline.sections if s.heading == "Constraints")
        for banned in problem.meta.banned_imports:
            if banned not in constraints:
                issues.append(
                    Issue(
                        "PROBLEM_BANNED_UNDOCUMENTED",
                        path,
                        f"Constraints must name the banned import {banned!r} (meta.yaml)",
                    )
                )
    return issues


def _check_hints(problem: ProblemPackage) -> list[Issue]:
    if problem.hints is None:
        return []
    headings = problem.hints.headings
    levels_ok = all(heading.startswith(f"Hint {n}") for n, heading in enumerate(headings, start=1))
    if len(headings) != 3 or not levels_ok:
        return [
            Issue(
                "HINTS_HEADINGS",
                str(problem.dir / "hints.md"),
                "hints.md needs exactly three H2 headings 'Hint 1 ...', 'Hint 2 ...', "
                f"'Hint 3 ...'; found {list(headings)}",
            )
        ]
    return []


def _check_python(problem: ProblemPackage) -> list[Issue]:
    meta = problem.meta
    language = meta.language if meta is not None else None
    entry = meta.entry if meta is not None and language == "python" else None
    issues: list[Issue] = []
    for name, facts in problem.python.items():
        path = str(problem.dir / name)
        if facts.syntax_error is not None:
            issues.append(Issue("PYTHON_SYNTAX", path, f"syntax error at {facts.syntax_error}"))
            continue
        if entry is not None and name in PYTHON_ENTRY_FILES and entry not in facts.defined:
            issues.append(
                Issue("PROBLEM_ENTRY_MISSING", path, f"{name} must define top-level {entry}")
            )
        if name == "starter.py":
            issues.extend(_check_starter(problem, path, facts, entry))
        if name.startswith("test_"):
            runtime = meta.runtime if meta is not None else None
            issues.extend(_check_test_module(name, path, facts, entry, language, runtime=runtime))
    return issues


def _check_starter(
    problem: ProblemPackage, path: str, facts: PythonFacts, entry: str | None
) -> list[Issue]:
    issues: list[Issue] = []
    if not facts.raises_not_implemented:
        issues.append(
            Issue("PROBLEM_STARTER_NOT_STUB", path, "starter.py must raise NotImplementedError")
        )
    solution = problem.python.get("solution.py")
    if entry is None or solution is None or solution.syntax_error is not None:
        return issues
    starter_sig = facts.signatures.get(entry)
    solution_sig = solution.signatures.get(entry)
    if starter_sig is not None and solution_sig is not None and starter_sig != solution_sig:
        issues.append(
            Issue(
                "PROBLEM_SIGNATURE_MISMATCH",
                path,
                f"starter {entry} takes {list(starter_sig)}; solution takes {list(solution_sig)}",
            )
        )
    return issues


def _check_test_module(
    name: str,
    path: str,
    facts: PythonFacts,
    entry: str | None,
    language: str | None,
    *,
    runtime: str | None,
) -> list[Issue]:
    issues: list[Issue] = []
    if language == "sql" and "submission.sql" not in facts.string_constants:
        issues.append(
            Issue(
                "PROBLEM_TEST_IMPORT",
                path,
                f"{name} must read the learner's query: Path(__file__).with_name('submission.sql')",
            )
        )
    if "solution" in facts.imported_modules:
        issues.append(
            Issue(
                "PROBLEM_TEST_SOLUTION_IMPORT",
                path,
                "tests must never import solution; they import the learner's code from submission",
            )
        )
    if entry is not None and entry not in facts.submission_names:
        issues.append(
            Issue(
                "PROBLEM_TEST_IMPORT",
                path,
                f"{name} must import the learner's code: from submission import {entry}",
            )
        )
    for module in sorted(facts.imported_modules):
        reason = _test_import_problem(module, runtime)
        if reason is not None:
            issues.append(
                Issue("TEST_IMPORT_NOT_ALLOWED", path, f"{name} imports {module!r}: {reason}")
            )
    minimum = PUBLIC_TESTS_MIN if name == "test_public.py" else HIDDEN_TESTS_MIN
    found = len(facts.test_functions)
    if found < minimum:
        issues.append(
            Issue(
                "PROBLEM_TEST_COUNT", path, f"{name} needs at least {minimum} tests; found {found}"
            )
        )
    return issues


def _test_import_problem(module: str, runtime: str | None) -> str | None:
    """Why a test may not import ``module``, or None when it may (CONTENT_SPEC section 6.4)."""
    for banned, reason in TEST_IMPORT_FORBIDDEN.items():
        if module == banned or module.startswith(banned + "."):
            return f"forbidden in tests: {reason}"
    top = module.split(".", 1)[0]
    if top == "solution" or top in sys.stdlib_module_names or top in TEST_IMPORT_ALLOWED:
        return None  # solution has its own code, PROBLEM_TEST_SOLUTION_IMPORT
    if top == "torch" and runtime == TEST_IMPORT_TORCH_RUNTIME:
        return None
    return (
        "not allowed in tests: use the stdlib, numpy, scipy, pandas, scikit-learn, statsmodels, "
        "duckdb, pytest, hypothesis or dogma_data (torch only for py-torch problems)"
    )


def _check_not_case(pkg: TopicPackage) -> list[Issue]:
    return [
        Issue(
            "TOPIC_CASE_FILES",
            _file(pkg, name),
            f"{name} is a case file; set kind: case in topic.yaml (and in the syllabus) to use it",
        )
        for name in CASE_ONLY_NAMES
        if (pkg.dir / name).exists()
    ]


def _check_case(pkg: TopicPackage) -> list[Issue]:
    case = pkg.case
    if case is None:  # the loader fills it for every kind: case package
        return []
    return [
        *_check_case_files(pkg, case),
        *_check_case_problems(pkg),
        *_check_debrief(pkg, case),
        *_check_reveals(pkg, case),
        *_check_exhibits(pkg, case),
        *_check_case_text(pkg, case),
    ]


def _check_case_files(pkg: TopicPackage, case: CasePackage) -> list[Issue]:
    present = {
        "brief.md": case.brief is not None,
        "reveals.yaml": case.reveals is not None,
        "twist.md": case.twist is not None,
        "debrief.md": case.debrief is not None,
    }
    return [
        Issue("CASE_FILE_MISSING", _file(pkg, name), f"{name} is missing")
        for name in CASE_FILES
        if not present[name]
    ]


def _check_case_problems(pkg: TopicPackage) -> list[Issue]:
    if (pkg.dir / "problems").exists():
        return [
            Issue(
                "CASE_HAS_PROBLEMS",
                str(pkg.dir / "problems"),
                "a case has no auto-graded problems; remove problems/ or make it kind: topic",
            )
        ]
    return []


def _check_debrief(pkg: TopicPackage, case: CasePackage) -> list[Issue]:
    if case.debrief is None:
        return []
    path = _file(pkg, "debrief.md")
    if case.debrief.headings != DEBRIEF_SECTIONS:
        return [
            Issue(
                "CASE_DEBRIEF_SECTIONS",
                path,
                f"H2 sections must be exactly {list(DEBRIEF_SECTIONS)} in order; "
                f"found {list(case.debrief.headings)}",
            )
        ]
    return [
        Issue("CASE_TEXT_EMPTY", path, f"debrief section {section.heading!r} is empty")
        for section in case.debrief.sections
        if not section.text.strip()
    ]


def _check_reveals(pkg: TopicPackage, case: CasePackage) -> list[Issue]:
    if case.reveals is None:
        return []
    path = _file(pkg, "reveals.yaml")
    count = len(case.reveals)
    if not REVEAL_MIN <= count <= REVEAL_MAX:
        return [
            Issue(
                "CASE_REVEAL_COUNT",
                path,
                f"a case needs {REVEAL_MIN}-{REVEAL_MAX} reveals; found {count}",
            )
        ]
    return [
        Issue(
            "CASE_EXHIBIT_MISSING",
            _file(pkg, reveal.exhibit),
            f"reveal {reveal.when_asked!r} points at a missing exhibit",
        )
        for reveal in case.reveals
        if reveal.exhibit is not None and not (pkg.dir / reveal.exhibit).is_file()
    ]


def _check_exhibits(pkg: TopicPackage, case: CasePackage) -> list[Issue]:
    issues: list[Issue] = []
    for entry in case.exhibits:
        path = _file(pkg, entry)
        if (pkg.dir / entry).is_dir():
            issues.append(
                Issue("CASE_EXHIBIT_TYPE", path, "exhibits/ must hold files, not folders")
            )
        elif not entry.lower().endswith(EXHIBIT_SUFFIXES):
            issues.append(
                Issue("CASE_EXHIBIT_TYPE", path, "exhibits/ holds PNG and CSV files only")
            )
    return issues


def _check_case_text(pkg: TopicPackage, case: CasePackage) -> list[Issue]:
    issues: list[Issue] = []
    for name, text in (("brief.md", case.brief), ("twist.md", case.twist)):
        if text is not None and not text.strip():
            issues.append(Issue("CASE_TEXT_EMPTY", _file(pkg, name), f"{name} is empty"))
    flagged = sorted({name for name, text in _case_texts(pkg, case) if _TODO.search(text)})
    issues.extend(
        Issue("CASE_TODO", _file(pkg, name), "TODO placeholder left in the text")
        for name in flagged
    )
    return issues


def _case_texts(pkg: TopicPackage, case: CasePackage) -> list[tuple[str, str]]:
    """Every authored text in the package, paired with the file it came from."""
    pairs: list[tuple[str, str]] = []
    if case.brief is not None:
        pairs.append(("brief.md", case.brief))
    if case.twist is not None:
        pairs.append(("twist.md", case.twist))
    if case.debrief is not None:
        pairs.extend(("debrief.md", section.text) for section in case.debrief.sections)
    if case.reveals is not None:
        for reveal in case.reveals:
            pairs.append(("reveals.yaml", reveal.when_asked))
            pairs.append(("reveals.yaml", reveal.answer))
    if pkg.lesson is not None:
        pairs.extend(("lesson.md", section.text) for section in pkg.lesson.sections)
    for card in pkg.cards or ():
        for key, value in card.model_dump().items():
            if key not in ("id", "type") and isinstance(value, str):
                pairs.append(("cards.yaml", value))
    return pairs


def _find_cycle(graph: Mapping[str, set[str]]) -> list[str] | None:
    """Return one cycle as ``[a, b, ..., a]`` if the directed graph has any, else None."""
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, set())):
            if state.get(nxt) == 1:
                return [*stack[stack.index(nxt) :], nxt]
            if nxt not in state:
                found = visit(nxt)
                if found is not None:
                    return found
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if node not in state:
            found = visit(node)
            if found is not None:
                return found
    return None
