"""The grounding eval (I-22): scripted nudges at levels 1 to 3, graded by hand, offline.

Every answer below is scripted, so the run is deterministic. The expected verdicts were written by
hand from the reviewed hints in ``REVIEWED_MD``. Each row names the fragments that must be held back
and the fragments that must survive. The last tests run one leaking answer through the whole service
with the fake provider. Limits of the heuristic are in ``dogma.tutor.ceiling``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from tests.unit.tutor.support import HINTS_TEXT, PROBLEM_ID

from dogma.content.loader import load_problem
from dogma.core.config import Settings
from dogma.tutor import service
from dogma.tutor.ceiling import FORMULA_NOTE, formulas, guard_ceiling, token_set
from dogma.tutor.gateway import Gateway
from dogma.tutor.hints import parse_reviewed_hints
from dogma.tutor.providers.fake import FakeProvider

REVIEWED_MD = """## Hint 1: concept

The projection is the shadow of `a` on the line through `b`, so it is a scalar multiple `c · b`
of `b`. Finding `c` is the whole job.

## Hint 2: formula / approach

`c = (a·b) / (b·b)`. This makes the residual `a - c·b` orthogonal to `b`. Validate that
`b·b != 0` before dividing.

## Hint 3: code shape

Convert both inputs, check the shapes, compute the two dot products, and return
`coefficient * b_vec` as a new array.
"""
PROBLEM = (
    "Implement `vector_projection(a, b)`, which returns the projection of `a` onto the line "
    "through `b`."
)
PROBLEM_WITH_FORMULA = (
    r"Implement `vector_projection(a, b)`. It returns $\dfrac{\mathbf{a}\cdot\mathbf{b}}"
    r"{\mathbf{b}\cdot\mathbf{b}}\,\mathbf{b}$."
)
REVIEWED = parse_reviewed_hints(REVIEWED_MD)


@dataclass(frozen=True)
class Row:
    name: str
    level: int
    reply: str
    must_drop: tuple[str, ...]
    must_keep: tuple[str, ...]
    changed: bool
    problem: str = PROBLEM


ROWS = (
    Row(
        "leak: hint 2 formula in code span",
        1,
        "Start from `c = (a·b) / (b·b)` and check the shapes [S1].",
        must_drop=("check the shapes", "c = (a·b)"),
        must_keep=(),
        changed=True,
    ),
    Row(
        "leak: hint 2 formula in LaTeX",
        1,
        r"The scalar is $c = \dfrac{a \cdot b}{b \cdot b}$, so compute it first.",
        must_drop=("so compute it first",),
        must_keep=(),
        changed=True,
    ),
    Row(
        "leak: residual span of hint 2",
        1,
        "The residual `a - c·b` must be orthogonal to `b` [S1].",
        must_drop=("residual",),
        must_keep=(),
        changed=True,
    ),
    Row(
        "leak: hint 3 code shape at level 1",
        1,
        "Return `coefficient * b_vec` as a new array [S1].",
        must_drop=("coefficient",),
        must_keep=(),
        changed=True,
    ),
    Row(
        "leak: hint 2 validation check",
        1,
        "Validate that `b·b != 0` before you divide [S1].",
        must_drop=("Validate",),
        must_keep=(),
        changed=True,
    ),
    Row(
        "kept: hint 1 formula stays",
        1,
        "The answer is a multiple `c · b` of `b` [S1].",
        must_drop=(),
        must_keep=("a multiple `c · b` of `b`",),
        changed=False,
    ),
    Row(
        "kept: plain prose",
        1,
        "Name the concept: a projection is the shadow of one vector on another [S1].",
        must_drop=(),
        must_keep=("Name the concept",),
        changed=False,
    ),
    Row(
        "kept: code names, not formulas",
        1,
        "Check `.ndim` and `.shape` before you divide [S1].",
        must_drop=(),
        must_keep=("Check `.ndim` and `.shape`",),
        changed=False,
    ),
    Row(
        "kept: hyphenated file name",
        1,
        "Your file is `p01-vector-projection` [S1].",
        must_drop=(),
        must_keep=("p01-vector-projection",),
        changed=False,
    ),
    Row(
        "kept: unrelated sum",
        1,
        "Add `a + c` to the running total [S1].",
        must_drop=(),
        must_keep=("running total",),
        changed=False,
    ),
    Row(
        "mixed: only the leaking sentence goes",
        1,
        "Name the concept [S1]. Then `c = (a·b) / (b·b)` gives the scale. "
        "Which line checks the zero vector?",
        must_drop=("gives the scale",),
        must_keep=("Name the concept", "Which line checks the zero vector?"),
        changed=True,
    ),
    Row(
        "level 2: hint 2 formula is within the ceiling",
        2,
        "Use `c = (a·b) / (b·b)` for the scale [S2].",
        must_drop=(),
        must_keep=("for the scale",),
        changed=False,
    ),
    Row(
        "level 2: hint 3 code shape is above the ceiling",
        2,
        "Return `coefficient * b_vec` as a new array [S3].",
        must_drop=("coefficient",),
        must_keep=(),
        changed=True,
    ),
    Row(
        "level 3: nothing is above the ceiling",
        3,
        "Return `coefficient * b_vec` as a new array [S3].",
        must_drop=(),
        must_keep=("coefficient",),
        changed=False,
    ),
    Row(
        "problem statement already shows the formula",
        1,
        "Compute `(a·b) / (b·b)` first [S1].",
        must_drop=(),
        must_keep=("Compute",),
        changed=False,
        problem=PROBLEM_WITH_FORMULA,
    ),
    Row(
        "limit: a fragment of a later formula is held back",
        1,
        "The dot product `a·b` matters here [S1].",
        must_drop=("dot product",),
        must_keep=(),
        changed=True,
    ),
    Row(
        "limit: words describing a formula are not judged",
        1,
        "The coefficient is a dot product over b dot b, in words [S1].",
        must_drop=(),
        must_keep=("in words",),
        changed=False,
    ),
)


@pytest.mark.parametrize("row", ROWS, ids=[row.name for row in ROWS])
def test_scripted_answer_is_held_to_the_ceiling(row: Row) -> None:
    shown = [hint.text for hint in REVIEWED.up_to(row.level)]
    above = [hint.text for hint in REVIEWED.hints if hint.level > row.level]

    body, changed = guard_ceiling(row.reply, shown=shown, above=above, problem=row.problem)

    assert changed is row.changed, body
    for fragment in row.must_drop:
        assert fragment not in body
    for fragment in row.must_keep:
        assert fragment in body


def test_a_formula_and_its_latex_form_are_the_same_token_set() -> None:
    inline = token_set("c = (a·b) / (b·b)")
    latex = token_set(r"c = \dfrac{a \cdot b}{b \cdot b}")

    assert inline is not None
    assert inline == latex
    assert token_set("(b·b) / (a·b) = c") == inline


@pytest.mark.parametrize("span", ["coefficient", "np.asarray(x)", "a, b, c", "b·b", "x"])
def test_spans_without_an_operator_or_with_too_few_tokens_are_not_formulas(span: str) -> None:
    assert token_set(span) is None


def test_formulas_are_found_in_every_span_kind() -> None:
    text = r"Inline `a + c * d`, display $$x - y - z$$, paren \(p / q / r\), and $m ≥ n ≥ o$."

    assert len(formulas(text)) == 4


def test_the_guard_leaves_neutral_text_alone_and_reports_no_change() -> None:
    text = "Think about what the shape of the output should be [S1]."

    assert guard_ceiling(text, shown=(), above=(REVIEWED_MD,), problem=PROBLEM) == (text, False)


def test_an_answer_with_nothing_left_has_an_empty_body() -> None:
    body, changed = guard_ceiling(
        "Use `y = x + x` here [S1].",
        shown=(),
        above=(HINTS_TEXT,),
        problem=PROBLEM,
    )

    assert changed is True
    assert body == ""


# -- the whole service, one scripted answer, the fake provider ------------------------------------


def _target(settings: Settings) -> service.Target:
    problem_dir = settings.content_dir / "tst" / "alpha" / "problems" / "p01-double"
    meta = load_problem(problem_dir).meta
    assert meta is not None
    return service.Target(problem_dir=problem_dir, problem_id=PROBLEM_ID, meta=meta)


def _refuse(_question: str) -> bool:
    raise AssertionError("no question is expected at this level")


def test_an_l1_answer_that_leaks_hint_two_is_stripped_by_the_service(settings: Settings) -> None:
    provider = FakeProvider(
        ["Each value is `y = x + x`, so check the loop [S1]. Which step is it?"]
    )

    answer = service.hint(
        settings,
        _target(settings),
        level=1,
        yes=False,
        ask=_refuse,
        gateway=Gateway(settings, provider),
    )

    assert answer.nudge is not None
    assert "y = x + x" not in answer.nudge.text
    assert "Which step is it?" in answer.nudge.text
    assert FORMULA_NOTE in answer.nudge.text
    sent = " ".join(message.content for message in provider.requests[0].messages)
    assert "Each output value" not in sent, "the model must not see Hint 2 at level 1"


def test_a_nudge_made_only_of_a_later_formula_is_held_back(settings: Settings) -> None:
    provider = FakeProvider(["Use `y = x + x` for each value [S1]."])

    answer = service.hint(
        settings,
        _target(settings),
        level=1,
        yes=False,
        ask=_refuse,
        gateway=Gateway(settings, provider),
    )

    assert answer.reviewed is not None
    assert answer.nudge is None
    assert answer.nudge_note is not None
    assert "held back" in answer.nudge_note
