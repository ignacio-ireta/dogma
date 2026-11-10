# Content Spec: topic packages

*The exact file formats for `content/`. The quality bar and review checklist are in
[ENGINEERING_STANDARDS §5](ENGINEERING_STANDARDS.md#5-content-standards-the-curriculum-is-code).
**Canonical example:** [`content/cml/vector-operations/`](../content/cml/vector-operations/).
When in doubt, copy its shape.*

---

## 1. Layout

```text
content/<track>/<topic-slug>/          # <track>/<topic-slug> == the id in curriculum/syllabus.yaml
├── topic.yaml
├── lesson.md
├── cards.yaml
├── interview.md
└── problems/
    └── p01-<problem-slug>/            # p01 is always the core problem (the session must-do)
        ├── problem.md
        ├── meta.yaml
        ├── starter.py | starter.sql
        ├── solution.py | solution.sql
        ├── test_public.py
        ├── test_hidden.py
        ├── hints.md
        └── fixtures/                  # optional: CSV / data files the tests load
```

Slugs: lowercase, `a-z0-9-`. Problem dirs are `pNN-slug`. Never renumber them.

## 2. `topic.yaml`

```yaml
id: cml/vector-operations                 # MUST equal the syllabus id
title: Vector operations                  # MUST equal the syllabus title
tier: A                                   # MUST equal the syllabus tier
requires: []                              # topic ids; MUST exist in the syllabus
aliases: [dot product, projection, inner product]   # extra search terms
objectives:                               # 2–4 observable outcomes ("Implement…", "Explain…")
  - Implement dot product, norm and projection with NumPy, handling the zero vector
minutes: {lesson: 12, practice: 20, min_viable: 15}
interview_angle: One-line description of how this shows up in interviews.
sources:                                  # >= 1; see §8 for formats
  - "book: Strang, Introduction to Linear Algebra, ch. 1 & 4.2"
status: draft                             # draft | verified | reviewed | retired
```

## 3. `lesson.md`

YAML front-matter followed by **exactly these H2 sections in this order**:

```markdown
---
title: Vector operations
minutes: 12
min_viable: "Read The system + Worked example, then do p01."
---

## The system
## Worked example
## Why it works
## Signal vs. noise
## Where it breaks
## Interview angle
## Connections
```

- Math in KaTeX: inline `$…$`, display `$$…$$`. Define every symbol and its shape at first use.
- Target 400–900 words. Prefer a table or decision list over a paragraph.
- Optional colour tags for later rendering: `{.input}` (blue), `{.latent}` (orange), `{.fitted}`
  (green), `{.error}` (red).

## 4. `cards.yaml`

```yaml
cards:
  - id: cml/vector-operations#1           # <topic id>#<n>, n never reused
    type: qa                              # qa | cloze
    front: What is the projection of a onto b?
    back: "proj_b(a) = (a·b / b·b) · b — a scalar multiple of b."
  - id: cml/vector-operations#2
    type: cloze
    text: "a·b = ‖a‖‖b‖ {{c1::cos θ}}"
```

4–8 cards per topic. Minimum information: one fact per card, no lists to memorise.

## 5. `interview.md`

```markdown
# Interview notes — <title>

## Q1. <question as an interviewer would ask it>
**60-second answer.** <calm, senior, trade-offs first>
**Follow-ups.** <2–3 likely follow-ups, each with a one-line answer>

## Q2. …

## Common mistakes
- …
```

2–4 questions. No personal names, employers or projects.

## 6. Problem files

### 6.1 `problem.md`

```markdown
# <Title>

<One-paragraph task statement.>

## Contract
- `arg`: type, shape, dtype, valid range
- Returns: type, shape, dtype
- Raises: `ValueError` when …
- Must not mutate inputs.

## Examples
<≥2 examples as code blocks with input → output>

## Constraints
- Allowed: …   Not allowed: … (mirrors meta.yaml banned_imports; "Allowed" stays inside the runtime, §6.2)

## Done looks like
<What passing means, and which edge-case families the hidden tests cover. Name the families, not the values.>
```

### 6.2 `meta.yaml`

```yaml
id: cml/vector-operations/p01-vector-projection
version: 1                     # bump when tests, tolerances or the contract change
title: Vector projection
language: python               # python | sql
runtime: py-numpy              # py-stdlib | py-numpy | py-data | sql-duckdb | py-torch | ...
entry: vector_projection       # python: function or class name; sql: "query"
difficulty: easy               # easy | medium | hard
minutes: 20
core: true                     # exactly one core problem per topic (p01)
tolerance: {atol: 1.0e-9, rtol: 1.0e-9}
banned_imports: [numpy.linalg] # modules the submission may not import or access
timeout_s: 30                  # whole test run
status: draft
```

The default environment supplies `py-stdlib`, `py-numpy`, `py-data` and `sql-duckdb`. For
`py-torch`, select the optional group on grading and verification commands:
`uv run --frozen --group torch dogma content verify <topic-id>`. Other runtime stacks in
the architecture remain planned; a runtime name in metadata does not install dependencies.

The runtime also sets what a submission may import (ADR-0021). `py-stdlib`: the standard library
only. `py-numpy`: plus NumPy. `py-data`: plus SciPy, pandas, scikit-learn and statsmodels.
`py-torch`: plus PyTorch. The static check rejects any other import before the code runs (rule
`runtime-import`); `banned_imports` narrows a runtime further, for a library that would do the
problem's job. Tests are not bound by the runtime: §6.4 lists what they may import.

### 6.3 Starter and solution

- **Python:** `starter.py` has the exact signature, type hints, a docstring restating the
  contract, and `raise NotImplementedError`. `solution.py` is readable reference code whose
  comments tie each step to the lesson's system.
- **SQL:** `starter.sql` contains a comment block with the task and the table schemas, plus
  `SELECT 1;`. `solution.sql` is the reference query. Use DuckDB SQL.

### 6.4 Tests: the harness contract

The runner stages these files for each attempt:

```text
submission.py | submission.sql   # the learner's file (or solution/starter during verification)
test_public.py  test_hidden.py   # tests
fixtures/                        # if present
```

The learner's file and `fixtures/` are copied into the attempt's writable run directory, and the
tests run from a protected sibling bundle that holds the same layout. Learner code cannot write
to the bundle, and the sandbox denies writes outside the run directory. Authors don't need to do
anything different. Your tests see the same files either way. The test runs as
`pytest -q --hypothesis-seed=0 test_public.py [test_hidden.py]` with the bundle as its test
directory. Two environment details matter to authors:

- Temporary files are welcome. `tmp_path`, `tempfile` and Hypothesis's `.hypothesis` database
  write inside the attempt. Writes anywhere else and all network access are denied, and
  `/dev/null` accepts data writes only (ARCHITECTURE §4.2).
- Symlinks inside `fixtures/` are rejected. Keep fixtures as plain files and directories.

Therefore:

- **Python tests import the learner's code with** `from submission import <entry>`.
- **SQL tests read** `Path(__file__).with_name("submission.sql").read_text()`, load fixtures into an
  in-memory DuckDB connection, and compare the result against an expected result computed
  independently (pandas over the same fixtures) or a small hard-coded table. Compare
  order-insensitively unless the task requires `ORDER BY`.
- Tests **never** import `solution.py`. They may only use the stdlib, numpy, scipy, pandas,
  scikit-learn, statsmodels, duckdb, pytest, hypothesis and `dogma_data` (§6.6), plus torch for
  `py-torch` problems when its optional dependency group is selected. Validation enforces this
  (`TEST_IMPORT_NOT_ALLOWED`) and also refuses `statsmodels.api` (a 3 s import: use
  `statsmodels.datasets.<name>` or a specific module), `datasets` and `huggingface_hub`.
- **Name each test after the behaviour it checks** (`test_zero_vector_raises_value_error`). The
  name is what the learner sees for a failing hidden test.
- Give assertions a message that names the *kind* of mistake (`err_msg="shape mismatch: …"`).
- `test_public.py`: ≥3 tests (the examples from problem.md + ≥1 edge case).
- `test_hidden.py`: ≥5 tests. It must include an **oracle comparison** on random inputs against
  an independent implementation (library call or a different formula) whenever one exists, plus
  edge cases: empty or degenerate input, dtype, large and small magnitudes, invalid input raises,
  inputs not mutated. Use Hypothesis for properties (`@settings(max_examples<=100, deadline=None)`).
- Deterministic: no unseeded randomness (`np.random.default_rng(<int>)`). Hypothesis is seeded by
  the runner.
- **Hidden failure output shows only the first meaningful line** of the message. That line must
  name the kind of mistake and must never contain hidden inputs, shapes or expected answers.
  Lessons from weeks 1–3:
  - Use `@settings(report_multiple_bugs=False, ...)` on Hypothesis tests (or route every check through
    one assertion helper). Otherwise several distinct failures surface as a `BaseExceptionGroup` header
    that names no mistake.
  - Give parametrized hidden tests neutral `ids=` (`ids=["empty", "single", "ties"]`). Default ids
    print the hidden inputs in the test name.
  - Replace bare `pytest.raises` with a small helper that names the mistake when nothing (or the
    wrong exception type) is raised.
  - Wrap learner calls whose failure would otherwise surface as a raw traceback (`AttributeError`,
    `TypeError`, `IndexError`) so the message says which contract rule was broken.
- `banned_imports`: a bare name (`random`, `bisect`) bans the *import* only. A dotted name
  (`numpy.linalg`, `scipy.stats.ttest_ind`) also bans attribute access to its last component.
- Fast: the whole run takes under 10 s on a laptop.

### 6.6 Test data

Bulk random data hides failures that real data exposes (overflow on large scores, gradient descent
on mixed scales, autocorrelation, ties from rounding) and teaches the wrong intuition (scaling
never matters, AUC is always 0.5). Where the algorithm's behaviour depends on the data's
structure, use the test-only package `dogma_data` ([DATASETS.md](DATASETS.md), ADR-0020).
Submissions can't import it: the static check rejects it in every problem.

**Choose the data by what the test proves:**

| Kind | Use it when | Oracle |
|---|---|---|
| Hand-built | An exact contract rule: empty, ties, overflow, collinear, invalid input | Literal values |
| Real (`dogma_data.load`, `load_series`, `load_frame`) | Behaviour depends on real structure: scales, spectra, autocorrelation, correlated features, rounding ties, skew, NaN | A library oracle on the same data, or a metamorphic invariant |
| Semi-synthetic | A known truth on realistic inputs: real `X`, then `y = X @ beta + noise` written inline | The known truth plus a library oracle |
| Realistic synthetic (`dogma_data.synth`) | No real set fits, or a property must be controlled: AUC, imbalance, tail weight, outlier channels | A library or reference oracle |

**Realism checklist.** Ask which of these the algorithm is sensitive to, and make sure at least
one hidden test exercises each that applies: features on scales spanning orders of magnitude;
strongly correlated or near-collinear features; heavy tails and outliers; uneven noise; rare
classes (1:100); scores that carry signal (AUC 0.7–0.95) and tied scores; values rounded to a few
decimals (ties, duplicates); NaN gaps; autocorrelation, season and trend; skewed users or
categories; dirty categorical keys (case, stray spaces).

**Robustness rules:**

- Compute expected values at runtime from an oracle on the same data. Never hard-code a number
  read off a random draw or a real dataset; literal expectations belong to hand-built cases.
- A statistical assertion states its bound (binomial or CLT, at least 4 standard errors).
- Seed data with `dogma_data.rng("<problem id>:<purpose>:<i>")`, not `range(8)`. The exercise
  audit varies the draw through `DOGMA_DATA_SALT`, so a test that passes only for one lucky draw
  is caught. Hypothesis stays seeded by the runner.
- Load real data in a module-scoped fixture, never at import time: a collection error would show
  as a hidden failure that names no mistake.
- Every realism test maps to a sentence in the Contract or Constraints of `problem.md`. If the
  behaviour is only implied, add the sentence (and bump the version) or report it; don't test it
  silently. Pick tolerances the conditioning allows: on longley (cond ≈ 4.9e9) compare fitted
  values or the objective, not coefficients at `rtol=1e-6`.
- Public tests stay literal (they mirror the examples). The one exception is a self-describing
  call such as `load("iris")`.
- Keep the whole run under 10 s with a naive correct solution, not only the reference. Slice
  large text (a pure-Python BPE on 1 MB times out).
- Name the dataset in a `problem.md` example and list the realism families under "Done looks
  like" (families, never values). Show one real-data failure in the lesson's "Where it breaks",
  with rounded numbers reproduced by a snippet (BLAS builds differ in the last digits).
- Prove each realism test earns its place: a hand-written realistic wrong solution must pass the
  old tests and fail the new ones (`scripts/exercise_audit.py realism`), and the references must
  survive `exercise_audit.py seeds`.

### 6.5 `hints.md`

```markdown
## Hint 1 — concept
## Hint 2 — formula / approach
## Hint 3 — code shape
```

Exactly three. Each builds on the last. None contains a complete solution. Headings may use `## Hint N: label` or `## Hint N — label`.
`dogma hint` shows these reviewed hints first (I-22), so their quality directly shapes the tutor.

## 7. Verification and the learner's loop (`dogma`)

```bash
uv run --frozen dogma content verify cml/vector-operations    # one topic (or one problem path)
uv run --frozen dogma content verify --all                    # every topic, plus repository checks
```

`content verify` prints every rule violation for each topic, then grades each problem in a scratch
directory. A problem passes when the solution passes every public and hidden test, twice with
identical outcomes, the starter fails at least one test, there are at least 3 public and 5 hidden
tests, and the solution has no static findings (no banned imports). It then runs `ruff check` on the
topic directory and exits 1 on any failure. A topic may be marked `status: verified` only when every
one of its problems verifies, `ruff check` on its directory is clean, and its YAML files parse.

A new topic starts from `dogma content new <topic id>`, which copies `templates/` with TODO
placeholders. The scaffold loads and passes the structural rules, but it does not verify until every
TODO is replaced and the problem directory is renamed from `p01-todo` to `p01-<slug>`.

**Authors and reviewers:** grade mutants and alternatives with `dogma submit <problem> --submission
<file>`. Runs with `--submission` are never recorded in the learner's progress (unless `--record` is
passed), so mutation testing can't mark topics solved.

The learner's own loop:

```bash
uv run --frozen dogma path                                    # this week's topics and their status
uv run --frozen dogma open   cml/vector-operations/p01-vector-projection
uv run --frozen dogma run    cml/vector-operations/p01-vector-projection
uv run --frozen dogma submit cml/vector-operations/p01-vector-projection
```

`open` copies the starter into `workspace/` and never overwrites your file. To start over (a timed
re-solve, say), `open --fresh` first keeps a copy under `var/workspace-history/`, then restores the
starter. `run` checks it against the public tests; `submit` also runs the hidden tests. Both exit 0
only when everything they ran passed.

## 8. Source formats (`topic.yaml: sources`)

- `"book: <Author>, <Title>, <chapter/section>"`: standard textbooks (Strang, ISLR/ESL,
  Goodfellow et al., Bishop, Sutton & Barto, Jurafsky & Martin, Murphy).
- `"paper: <Authors> <Year>, <Title> (arXiv:<id>)"`
- `"library: <path under sources/library/>"`: an imported note.
- `"source: <path under sources/>"`: any other imported file (e.g. `interview-prep/...`).
  `library:` and `source:` paths must exist; validation reports `SOURCE_NOT_FOUND`.
- `"corpus: <source-id>/<file> p.<page>"`: once the corpus exists.

Paraphrase. Never copy text from deep-ml or other problem sites.

No scaffold placeholder may survive: validation reports `TOPIC_TODO` for any `TODO` left in a
topic's authored files (Markdown, YAML, Python, SQL; fixtures excluded) and for a `pNN-todo`
problem directory. Case packages have the equivalent `CASE_TODO`.

## 9. Case packages

A **case** is a system-design or product case, practised in an interview: the candidate gets a
brief, asks questions, and the interviewer releases facts only for good questions. Cases live in the
local `sys` track. A case package is a topic package with `kind: case`. It has **no problems**, so
nothing is auto-graded. Its quality is checked by `dogma content verify` (validation and
`ruff check` on the directory). There is no runner.

```text
content/sys/churn-prediction-case/
├── topic.yaml        # kind: case  (the syllabus entry must say kind: case too)
├── lesson.md         # the same seven H2 sections as §3, in the same order
├── cards.yaml        # 4-8 cards, as §4
├── interview.md      # optional, as §5 (a case may have none)
├── brief.md          # candidate-facing setup: what the interviewer reads aloud
├── reveals.yaml      # 2-8 reveals, released only when asked
├── twist.md          # the new fact dropped after the candidate's first structure
├── debrief.md        # the strong answer's outline, one H2 per RASS part
└── exhibits/         # optional: PNG or CSV files only, referenced from reveals
```

`topic.yaml`: the same fields as §2, plus `kind: case`. `kind` defaults to `topic`, so existing
topics are unchanged. The syllabus entry carries `kind: case` as well, and the two must agree
(`TOPIC_KIND_MISMATCH`). Tier, title and id rules are the same as for topics.

**brief.md** is free Markdown with a `# <title>` line. It holds no hints about the answer.

**reveals.yaml** is a YAML list (not a mapping) of reveals:

```yaml
- when_asked: How many customers churned last quarter?   # non-empty; the trigger, in words
  answer: Thirty of the two hundred.                      # non-empty; the fact, with numbers
  exhibit: exhibits/counts.csv                            # optional; exhibits/<name>.png|csv
```

Unknown keys are errors. Matching a question to a reveal is done by the mock runner (I-18c), not
here.

**debrief.md** has exactly these H2 sections, in this order, one per RASS part of the mock rubric:
`Reframe`, `Ask`, `Structure`, `Synthesize`. Every section has text.

**exhibits/** holds flat files only, each a PNG or a CSV. Folders and other file types are errors.

**Rules** (issue codes from `dogma.content.validate`):

| Code | Rule |
|---|---|
| `CASE_FILE_MISSING` | `brief.md`, `reveals.yaml`, `twist.md` and `debrief.md` must all exist |
| `CASE_HAS_PROBLEMS` | a case has no `problems/` directory |
| `CASE_DEBRIEF_SECTIONS` | `debrief.md` H2 sections are exactly the four RASS parts, in order |
| `CASE_TEXT_EMPTY` | `brief.md`, `twist.md` and each debrief section contain text |
| `CASE_REVEAL_COUNT` | 2 to 8 reveals |
| `CASE_EXHIBIT_MISSING` | a reveal's `exhibit` file exists |
| `CASE_EXHIBIT_TYPE` | `exhibits/` holds only PNG or CSV files, no folders |
| `CASE_TODO` | no `TODO` placeholder left in any authored text (scaffolds fail until filled) |
| `TOPIC_KIND_MISMATCH` | `topic.yaml` kind equals the syllabus kind |
| `TOPIC_CASE_FILES` | a `kind: topic` package has no case files (`brief.md`, `reveals.yaml`, `twist.md`, `debrief.md`, `exhibits/`) |
| `CONTENT_SCHEMA_INVALID` | `reveals.yaml` fails its schema (unknown keys, empty fields, bad `exhibit` path) |

The lesson, cards and interview rules of §3 to §5 apply to cases unchanged. Interview is optional for
a case; if `interview.md` exists, its rules apply.

**Scaffold:** `dogma content new sys/<case>` uses `templates/case/` when the syllabus entry is
`kind: case`, and `templates/topic/` otherwise. The case scaffold loads, and fails verification with
`CASE_TODO` only, until the TODOs are filled.

**Coverage:** a case counts as authored content: its package, lesson and cards appear in the
coverage report, and `cases` / `with_cases` count case topics per track and tier. Its `Problems`
count is zero by design.
