# Architecture

*A local-first, verified-content practice platform for ML / DS / DL / RL / LLM engineering. Status: design approved; built in small increments tracked in
[ROADMAP](ROADMAP.md). Keep this document current. Decisions that change it get an ADR (§9).*

---

## 1. What we're building

A local practice platform: **topic lessons + auto-graded coding problems + spaced review + mock
interviews**, all driven by a learner model that decides what to do next, with a tutor that runs
on your local models and cites your own library.

**Goals**

1. Develop foundations and practical interview skills for data science and AI engineering,
   then extend them through optional advanced topics.
2. Every problem is **verified**: the reference solution passes, the starter fails, and the
   tests check against an independent library implementation (an "oracle"), not the author's
   own reasoning.
3. **The learner never has to plan.** Opening the app shows today's session: its shape, its
   length, its first action.
4. Measure *readiness* (can you do it from memory, under time pressure, and explain it), not
   just completion.
5. Private and offline by default. The repo can be published as a portfolio piece without
   leaking anything (ADR-0011, ADR-0016).
6. **Self-contained.** Everything the platform needs lives in this repo. Only the AI runtime
   (model weights, inference servers) is external (ADR-0014).

**Non-goals:** multiple users, accounts, cloud hosting, leaderboards, streaks, mobile apps,
reproducing deep-ml's problems, a CMS.

## 2. Scope vs. time: the budget

**Default course configuration:** **3 × 45-minute sessions per study day**, six study days
per week, balanced DS / AI engineering, and content reviewed by a **second agent**. The dates
below describe the configured course calendar, not a personal availability or employment goal.

- **Calendar:** 14 weeks (Mon 26 Oct 2026 → Sun 31 Jan 2027) ≈ 78 study days after holidays ≈
  **234 sessions**. These are the authored dates. The learner can move them by whole weeks
  (time-warp, §4.4, ADR-0019); `core.yaml` is never rewritten.
- **Allocation:** 2 Learn sessions per day (~150), 1 Consolidate per day (~70), one Interview
  session per week from week 3 (taken from Consolidate), and the **final week is interview-only**.
- All 545 topics (538 outline topics plus the 7-topic local `sys` track) would need 400+ hours. **Tier A = 164 topics**, which fits about 150 Learn
  sessions because small topics share a session.

| Tier | Meaning | Rule of thumb |
|---|---|---|
| **A: core** | Senior DS / AI-eng interviews assume it | You'd be embarrassed not to implement or explain it on a whiteboard |
| **B: differentiator** | Marks you as senior or specialised | The optional pool after the core path |
| **C: specialist** | Niche, research-frontier, or hardware-bound | Backlog, only if a target role needs it |

Actual split (`curriculum/syllabus.yaml`). **Tier A is exactly `curriculum/paths/core.yaml`:**
13 themed weeks, then an interview week.

| Track | Topics | A | B | C |
|---|---:|---:|---:|---:|
| cml: Classical ML | 111 | 56 | 44 | 11 |
| dl: Deep Learning | 88 | 34 | 38 | 16 |
| llm: Large Language Models | 115 | 25 | 44 | 46 |
| gen: Diffusion & World Models | 28 | 0 | 17 | 11 |
| rl: Reinforcement Learning | 66 | 5 | 31 | 30 |
| gnn: Graph NNs | 11 | 0 | 9 | 2 |
| inf: Inference Engineering | 46 | 12 | 20 | 14 |
| gpu: CUDA & Triton | 11 | 0 | 3 | 8 |
| mlc: ML Compilers | 10 | 0 | 0 | 10 |
| safe: AI Safety | 6 | 1 | 4 | 1 |
| sql: SQL & pandas | 15 | 15 | 0 | 0 |
| algo: Algorithms | 19 | 12 | 5 | 2 |
| robo: Robotics | 2 | 0 | 1 | 1 |
| math: Math drills | 10 | 4 | 5 | 1 |
| sys: ML System Design & Cases (local) | 7 | 0 | 7 | 0 |
| **Total** | **545** | **164** | **228** | **153** |

**A gap in deep-ml we close ourselves:** the syllabus has no *ML system design*, product sense or
project-narrative track, and senior loops weigh those heavily. We add a local track
`sys: ML System Design & Cases` (`origin: local`), seeded from the imported case scenarios, the RASS
framework and the scoring rubric: design a churn / fraud / ranking / RAG system, then defend metrics, data,
serving, monitoring and trade-offs. Case questions use authored synthetic scenarios rather than
private work histories.

## 3. System overview

```text
                     ┌─────────────────────────── repo (git) ───────────────────────────┐
                     │  curriculum/syllabus.yaml   content/<track>/<topic>/…   docs/     │
                     └───────────────┬──────────────────────────────────────────────────┘
                                     │ load + validate (Pydantic) → compiled catalog
┌──────────────┐   HTTP /api/v1   ┌──▼───────────────────────── dogma (Python 3.12) ─────────────┐
│ web/ (React) │◄────────────────►│ api ── learning ──┬── content ── grading ── runner ──► subprocess
│  Today       │                  │ (FastAPI)  (planner,│                       (sandbox)    dogma_harness
│  Problem     │   ┌──────────┐   │            FSRS,    └── tutor ── corpus (FTS5/BM25)          │
│  Review/Map  │   │ CLI      │◄──┤            mastery)        │                                 │
│  Interview   │   │ (Typer)  │   └────────────┬───────────────┼─────────────────────────────────┘
└──────────────┘   └──────────┘                │               │ OpenAI-compatible, loopback only
                                   var/dogma.db (SQLite)       ▼
                                   var/runs/, var/llm_calls.jsonl   llama-server router 127.0.0.1:8080
                                                                    (qwen3.8-27b-q6, External SSD)
     committed: sources/ (scrubbed seed snapshot)
     gitignored: private/ (personal material)  corpus/ (papers→md)  var/  workspace/
```

Data flows one way, from content to the learner. The learner's state lives only in
`var/dogma.db`.

## 4. Components

### 4.1 Curriculum & content model

**Curriculum-as-code** (ADR-0002). The syllabus is the coverage spine. Content is a tree of topic
packages. Both are validated against Pydantic schemas and compiled into one catalog at startup.

```text
content/
└── cml/
    └── gradient-descent/                 # dir name = slug part of the syllabus id
        ├── topic.yaml                    # metadata (below)
        ├── lesson.md                     # fixed section order (ENGINEERING_STANDARDS §5.3)
        ├── cards.yaml                    # spaced-repetition cards (Q/A, cloze)
        ├── interview.md                  # questions, 60-second answers, follow-ups, failure modes
        └── problems/
            └── p01-batch-gd/
                ├── problem.md            # statement, I/O contract, examples, constraints
                ├── meta.yaml             # id, version, runtime, difficulty, minutes, tolerances, banned imports
                ├── starter.py            # signature + docstring; runs; fails tests
                ├── solution.py           # reference; readable; explains the system in comments
                ├── test_public.py        # shown to learner (examples + 1–2 edge cases)
                ├── test_hidden.py        # edge cases, property tests, oracle comparisons
                └── hints.md              # 3-step hint ladder: concept → formula → code shape
```

`topic.yaml`:

```yaml
id: cml/gradient-descent            # must exist in syllabus.yaml (or a local track)
tier: A
requires: [cml/partial-derivatives-and-gradients, cml/vector-operations]
objectives:                          # observable outcomes, used by the Close step
  - Implement batch gradient descent for linear regression with a convergence criterion
  - Explain why the learning rate must be below 2/L for a quadratic with curvature L
interview_angle: "Why does GD diverge? How would you pick a learning rate in practice?"
minutes: {lesson: 15, practice: 25, min_viable: 15}   # min_viable = low-energy path
sources: [library:AI-ML/03-Optimization/Cauchy Steepest Descent 1847.md, corpus:library-papers/1412.6980]
status: draft | verified | reviewed | retired
```

**Content types**: `lesson`, `problem` (auto-graded), `lab` (guided work against real systems,
e.g. measuring TTFT/ITL on your own llama-server; checked from the artifacts it produces),
`cards`, `interview` (written or spoken answers, scored against a rubric), `case` (sys-track
system design).

**Compiled catalog**: `dogma catalog build` validates content and writes `var/catalog.json`
(topics, problems, prereq DAG, coverage). The API reads that file when valid, otherwise builds
a catalog from local content without writing it. Rebuild the compiled catalog after content
changes. Responses use learner-facing schemas; the API never serves solution or test files.

**Coverage**: `dogma content coverage` reports, per syllabus topic, whether it has a lesson,
problems, cards and an interview, plus its status.

### 4.2 Runner (sandboxed execution)

Executes learner code locally (ADR-0006). The threat model is a single trusted user. The sandbox
exists to contain *runaway* code (infinite loops, memory blowups, stray writes), not an
adversary.

- **Per attempt:** `var/runs/<attempt_id>/` is the only writable location and is retained. It
  holds `submission.py` (or `submission.sql`) and the fixtures, but no test files. The tests run
  from a **protected bundle**, a sibling `var/runs/.<attempt_id>.tests-<hex>/` outside the write
  allowance. It holds the selected test files, a snapshot of the submission and the fixtures,
  sealed read-only, and it is removed on every exit path, including timeouts, kills and staging
  failures. The attempt directory is kept (`src/dogma/runner/staging.py`). Pytest imports
  `submission` and reads fixtures beside `__file__` in the bundle, so the content contract is
  unchanged (CONTENT_SPEC §6.4). The runner spawns the harness as
  `python -P -s -m dogma_harness … --tests-dir <bundle>`. It doesn't use `-I`, because that
  silently ignores `PYTHONHASHSEED` (measured). The child gets: its own process group; a
  wall-clock timeout (`meta.yaml: timeout_s`, default 30 s); an RSS watchdog (psutil, 2 GB default,
  because `RLIMIT_AS` is unreliable on macOS); a disk watchdog (512 MiB growth of the run dir); an
  environment rebuilt from scratch (`PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`,
  `OMP_NUM_THREADS=4`, `HOME`/`TMPDIR` = run dir, so pytest's `tmp_path`, `tempfile` and
  Hypothesis's database work); cwd = run dir; and learner stdout captured by the harness with a
  64 KB cap. Statically rejected attempts never execute.
- **OS enforcement (I-21, `src/dogma/runner/policy.py`):** every harness launch is wrapped in
  `/usr/bin/sandbox-exec` with a fixed Seatbelt profile. The profile denies all network access and
  all file writes, then allows writes beneath the canonical attempt directory (passed as a
  parameter, never interpolated into the profile) and data writes to `/dev/null` alone. Pytest's
  logging plugin opens `/dev/null` at startup. Terminals and every other path stay denied. Reads
  are unrestricted. Child processes inherit the same profile, and the profile is applied before
  any learner code starts.
- **Supported platform (fail closed):** macOS only, with a working `sandbox-exec`. Elsewhere, or
  when the executable is missing or refuses the profile, learner execution raises
  `RUNNER_CRASHED` before any run directory is created, or immediately after the refusal with
  nothing executed. The attempt is never retried without the wrapper, and no unsandboxed fallback
  exists. Static rejections still return their normal report without launching anything. This is a
  deliberate narrowing of ADR-0006: execution needs a working OS sandbox, not merely a platform
  name.
- **Known limits (one trusted user, same account):** reads are not confined, so learner code can
  read same-user files, including the repo, `private/` and the test sources. Relocating the tests
  protects their integrity against learner writes. It does not hide them. The child shares the
  harness's pytest process and report channel, so a learner can forge its own report. Process-group
  cleanup does not contain a child that leaves its session. Hostile-code isolation would need a
  different architecture and is not claimed.
- **Static checks before running:** syntax (AST parse), required signature present, banned
  imports (e.g. no `sklearn` in from-scratch problems), optional "no Python loops" rule for
  vectorisation problems.
- **Runtimes** (`meta.yaml: runtime`):

| Runtime | Stack | Used by | Status |
|---|---|---|---|
| `py-stdlib` | the standard library only | algo and arithmetic problems | Phase 0 |
| `py-numpy` | + numpy | most cml/dl/llm/rl/inf/math problems | Phase 0 |
| `py-data` | + scipy, pandas, scikit-learn, statsmodels | oracles, pandas problems | Phase 0 |
| `py-torch` | + torch (MPS/CPU) | framework-level dl/llm | Optional `torch` group; I-19 acceptance pending |
| `py-jax` | + jax (CPU) | `dl/jax-*`, `dl/jit-and-vmap` | Phase 4 |
| `py-tinygrad` | + tinygrad | `dl/tensors-in-pytorch-and-tinygrad`, mlc track | Phase 4 |
| `sql-duckdb` | DuckDB on fixture tables | sql track | Phase 1 |
| `cuda-sim` | numba CUDA simulator (CPU) | gpu CUDA topics | **spike** |
| `triton-shim` | small numpy implementation of `tl.program_id/arange/load/store` | gpu Triton topics | **spike** (Triton has no macOS wheels) |
| `py-rl` | + gymnasium | rl track | Phase 4 |

  The stack is also what a submission may import (ADR-0021): the static check rejects any import
  outside the problem's runtime (rule `runtime-import`), and `banned_imports` narrows it further.
  All runtimes share one uv-managed environment with optional dependency groups, so the harness
  imports only what a problem needs. Currently only `torch` is an optional runtime group. Use
  `uv run --frozen --group torch dogma …` (including `serve` for browser grading); the runner
  reports `RUNNER_RUNTIME_UNAVAILABLE` when a `py-torch` problem needs the absent package. Other
  future stacks in this table are not installed by the default sync.

### 4.3 Grader & harness

`dogma_harness` (`src/dogma_harness/`) runs inside the sandbox. It depends only on the stdlib
and pytest, and never imports the app (import-linter contract). Content tests follow
[CONTENT_SPEC §6.4](CONTENT_SPEC.md): the learner's file is `submission.py` or `submission.sql`
in the protected bundle beside the tests (§4.2), and the tests import or read it (ADR-0015).
`--tests-dir` names that bundle. Without it the harness reads tests from `--run-dir`, as the
standalone CLI always has. Test paths are absolute, and the pytest rootdir is the test directory,
so visibility stays keyed to the file name. The harness runs
pytest programmatically with a small plugin that records each test's outcome, then writes
`report.json`. Learner `print`s are captured separately (truncated at 64 KB).
`scripts/grade.py` remains the legacy practice script. The supported CLI and quality gate use
`dogma.runner` and `dogma content verify`.

```text
GradeReport
  attempt_id, problem_id, problem_version, runtime
  status: passed | failed | error | timeout | oom | rejected (static check)
  tests[]: name, visibility (public|hidden), outcome (pass|fail|error), message, duration_ms
  static[]: rule, message, line
  stdout (truncated), duration_ms
```

**Kinds of check** (combine as needed):

- **I/O cases** with declared tolerances (`atol`/`rtol` in `meta.yaml`), shape and dtype checks.
- **Property tests** (Hypothesis): softmax rows sum to 1 and are shift-invariant, a projection is
  idempotent, a covariance is PSD…
- **Oracle comparisons** on random inputs: numpy.linalg, scipy.stats, sklearn, statsmodels,
  `torch.nn.functional`, DuckDB. This is what keeps LLM-authored content honest.
- **Gradient checks** (finite differences) for every backprop and autograd problem.
- **Result-set equality** for SQL, ignoring order unless the problem requires `ORDER BY`.
- **Test data** from `dogma_data` (content tests only): real offline datasets, vendored
  snapshots and seeded synthetic generators, so bulk random inputs have real structure. Tests
  read vendored files and local bundles, never the network. Each dataset and generator is
  listed in [DATASETS.md](DATASETS.md).

**Feedback contract** (ported from MVS `selfcheck`): blank or unattempted → a gentle reminder,
not a failure. Wrong → a precise message that names the *type* of error ("shape mismatch",
"values off by a constant factor", "fails only on empty input"). Right → a short OK. Public
test feedback may include inputs and expected values. Hidden feedback uses a neutral test name
and a filtered summary that omits hidden inputs, shapes, expected values and Hypothesis examples.

**The LLM never decides correctness** (ADR-0007). The tests do.

### 4.4 Learning engine

The learning engine is deterministic and explainable: every recommendation
comes with a one-line reason.

**Mastery states** per topic:

The current engine records `introduced` at Learn-session close and `solved` after a passing
core submit. It schedules FSRS card reviews, and since I-13a it runs timed re-solve windows that
earn `retained`. Since I-18a a learner answers a topic's interview question (`dogma answer`) and
self-scores it on `curriculum/rubrics/topic-answer-v1.yaml`. A `retained` topic becomes
`interview-ready` when one such answer scores 3 or higher on every dimension (`ready_at` is set).
Answers never lower mastery. A failed timed re-solve or a card lapse drops `interview-ready` back
to `consolidating`, and a later passing timed re-solve re-retains it. Mock scores (I-18c) are stored as `mock` answers linked to a mock session
and never change mastery: a mock records how you did, it does not move a topic.
The browser side (I-13b) starts a timed re-solve from Today, shows the budget countdown and asks
before a hint counts against the window. The CLI, the API and the browser share one set of rules.

```text
unseen → introduced → solved → consolidating → retained → interview-ready
          (lesson +     (all     (cards in SRS;   (timed re-solve,   (retained + answer
           predict)     tests    ≥1 good review    no hints, ≥7 d     rubric ≥3/4 on every
                        pass)    after ≥3 days)    after solving)     dimension)
```

A lapsed card or a failed timed re-solve drops the topic back to `consolidating`. Nothing is
ever reset to zero.

**Scheduling:** cards use FSRS (`fsrs` package, target retention 0.90, ADR-0010). Problems are
scheduled for timed re-solves with growing gaps of about 7, 21 and 60 days between passes
(so roughly days 7, 28 and 88 after the first solve), which turns understanding into habit:
repeated retrieval supports recall under time constraints.

Timed re-solves are **derived, not stored** (`dogma.learning.resolves`). A core problem's first
passing submit starts its schedule at stage 0, due 7 days later. Each `passed` window raises the
stage, capped at 3, and the next due time is 21, 60, then 120 days (maintenance) after the last
passed window closed. After a `partial` or `failed` window the problem is due 3 days after that
window closed, at the same stage. An `abandoned` window and a window still open are ignored by the
schedule. A window open for more than 12 hours is abandoned the next time windows are read.

A window opens with `dogma resolve <problem>`: the budget is the problem's `meta.minutes`, and the
starter is restored with the old file kept under `var/workspace-history/`. The first submit of
that problem decides it. All tests pass within the budget with no hints is `passed`. All tests
pass but late or hinted is `partial`. Anything else, including a rejected run, is `failed`. `run`
never decides a window. A hint or debug during a window asks first and counts against it.
Mastery: a `passed` window at least 7 days after solving moves `consolidating` to `retained`, and
a `failed` window drops `retained` back to `consolidating`, never lower. A topic still at `solved`
becomes `retained` only after a card review moves it to `consolidating`.

**Session planner** (`dogma today`, the Today page). The unit is a **45-minute session**. A
normal day is three of them. Inputs: due cards, path order, prerequisite DAG, topic states,
energy mode, weak spots from recent mocks. Each session type has a fixed internal structure, so
the structure stays predictable while the content changes (adapted from the MVS 5-stage loop):

| Session type | Blocks | When |
|---|---|---|
| **Learn** | Retrieve 5' (due cards, 3–7-day-old material first) → Predict 3' (pre-test on the new topic) → Learn 12' (lesson) → Practice 20' (core problem) → Close 5' (60-second answer, new cards enter SRS, transition note) | Sessions 1–2 of a normal day. Two small topics can share one Learn session |
| **Consolidate** | Review 15' (FSRS due cards) → timed re-solves 25' (1–2 older problems, interleaved) → Close 5' | Session 3 of a normal day |
| **Interview** | Mock block 40' (live coding + ML theory + a sys-track case, scored with the mock rubric) → debrief 5' | Replaces the first Consolidate of each week from path week 3. Every session of path week 14 (the interview week). Never in low-energy mode |
| **Minimum viable** | Review 10' → one familiar re-solve 10' | Any session, on low-energy days |


- **One must-do.** The first Learn session's core problem is the day's single required outcome.
  Everything else is a bonus.
- **Timed re-solves** (I-13a): Consolidate's 25' block takes due core re-solves, open windows
  first, then lowest stage, then earliest due. At most two whose budgets fit the block, never a
  topic planned earlier today. With none due, the block is familiar practice labelled untimed.
  Minimum-viable sessions are always untimed.
- **The first action is pre-chosen** ("open `cml/gradient-descent/p01`, read example 1"), and a
  10-minute forced start counts as a successful day.
- **Transition note** at close: where I stopped, the exact next step, any open blocker. It's
  shown at the top of the next session, so you never start by reconstructing where you were.
- **Low-energy mode:** any session can become a Minimum-viable session. No new material. Each
  topic's `min_viable` path covers days in between.
- **Missed days are absorbed:** overdue reviews are capped per day and spread out again. They
  never show up as a debt counter.
- **Time-warp** (I-30, ADR-0019): the week number, the weekly mock from week 3 and the
  interview-only final week follow a personal calendar, the authored dates moved by a whole-week
  offset (`calendar_warp`). `dogma warp` makes this week the week of the furthest core-path
  exercise the learner has run (any recorded attempt; week 1 with none), or a chosen week, or
  moves ahead or back, or restores the authored dates. Each move is previewed first. `dogma course
  reset` backs up, clears all progress, sets workspace files aside and makes this week week 1.
  Topic order never depended on the date, and card and re-solve due dates do not move: they
  measure memory over real time.
- **New topics** are introduced only when every `requires:` topic is ≥ `solved`, in path order
  (`curriculum/paths/*.yaml`) within the chosen tier.
- **Weekly mock** (from week 3, I-18c): `dogma mock` runs coding + ML theory + case, scored with the
  30-item `mock` rubric (`curriculum/rubrics/mock-v1.yaml`). The weakest part over the last three
  scored mocks is named in the Interview session's notes. No score trend, streak or backlog is shown.

### 4.5 Tutor

Every model call goes through an LLM gateway (ADR-0008):

```python
class LLMProvider(Protocol):
    def complete(self, req: CompletionRequest) -> CompletionResult: ...
    def stream(self, req: CompletionRequest) -> Iterator[CompletionChunk]: ...
```

Implemented providers: `local` (httpx → `http://127.0.0.1:8080/v1`; non-loopback URLs are
rejected) and `fake` (tests). External providers remain planned; `DOGMA_ALLOW_EXTERNAL` alone
does not enable one. Any future external provider must require both that setting and an
explicit per-request opt-in.

`hint`, `debug`, `score` and `interviewer` are implemented. `explain` and `probe` are planned.

| Mode | Trigger | Behaviour |
|---|---|---|
| `hint` | Learner asks | L1 concept → L2 formula/approach → L3 code shape: authored hint first, then an optional model nudge. L4 walkthrough requires two recorded failed submits or explicit confirmation; there is no elapsed-time unlock |
| `debug` | After a failing GradeReport | Points to the first wrong concept or line and names the error type. Doesn't rewrite the code |
| `explain` | Concept question | Answers grounded in the lesson + corpus, with citations (planned) |
| `probe` | After passing | Interviewer follow-ups: complexity, scale, failure modes, "what if the data doesn't fit in memory?" (planned) |
| `score` (I-18d) | `dogma answer --model-score`, `dogma mock score --model-score` | Advisory 1–4 scores per criterion (topic-answer dimensions or the 30 mock items), each with a quote that must occur verbatim in the learner's text. Items that break a rule (unknown id, value outside 1–4, quote not found, extra keys, duplicates) are dropped and counted. The reply is strict JSON, and no model prose is printed. Stored in `answer.model_score`; never changes the self-score or mastery. A topic answer given aloud has no text, so it is not scored. A mock is scored from its item notes only |
| `interviewer` (I-18d) | `dogma mock ask "<question>"` | The model sees the case brief and the numbered questions of the reveals, never the reveal texts or `debrief.md`. It replies `{"reveal": n}` or `{"reveal": null}`. The printed text is the reveal's own answer from `reveals.yaml`. A pick is kept only when the question shares a keyword with that reveal's question. Down model or unusable reply: the I-18c keyword match answers, with a note |

Model replies for `score` and `interviewer` are checked by `tutor/replies.py`, `tutor/score.py` and
`tutor/interviewer.py`. Both prompts are versioned (`score.md` v1, `interviewer.md` v1) with golden
snapshots. Every call is logged as metadata in `llm_calls.jsonl`, with mode `score` or `interviewer`.
New error codes: `TUTOR_SCORE_REPLY_INVALID`, `TUTOR_INTERVIEWER_REPLY_INVALID`,
`TUTOR_QUESTION_INVALID`, `ANSWER_NO_TEXT`.

**Tone contract** (from the pedagogy): a co-pilot, not an evaluator. Errors are data. Always name
what *kind* of difficulty it is (shape bug, concept gap, or just noise). Concise. No empty praise.

**Context assembly** (kept under about 12K estimated tokens for latency): versioned system
prompt + problem + learner code + filtered GradeReport + authored hints up to the requested
level + lesson excerpts + up to four corpus passages, at most one per source document. Current
context assembly does not read `private/`. Prompts live in `src/dogma/tutor/prompts/` and have
snapshot tests.

**Degradation:** if the model server is unavailable, authored hints at levels 1–3 still appear.
Only model-generated nudges, walkthroughs and debug answers need the server; grading and study
features continue to work.

### 4.6 Corpus (retrieval)

```text
sources/ (notes) ─┐
content/ (lessons)─┼─► chunk (heading/page-aware, ~500 tokens, keeps source+page+heading path)
docmd → corpus/md ─┘        └─► SQLite FTS5 (bm25, porter) = corpus/index.db ─► search(query, topics?, k)
```

- `dogma corpus sync` runs the vendored docmd
  (`uv run --frozen --project tools/docmd docmd sync corpus/raw/<id> --out corpus/md/<id>`,
  incremental by sha256), then re-chunks and re-indexes only what changed. OCR is off by default;
  `--ocr auto` or `--ocr force` opts in to OCR and, on macOS, the vendored project's locked OCR
  extra. `--no-convert` refreshes the index from existing local text without running docmd.
- **Topic tagging:** a keyword dictionary built from syllabus titles plus `aliases` in
  `topic.yaml`, optionally refined offline in batches by the local model.
- **Citations** look like `[library-papers/1412.6980 p.3]` and link to the converted Markdown.
- **Upgrade path** (ADR-0009): if BM25 recall falls short on a 30-query eval set, add an
  embedding model with hybrid reciprocal-rank fusion. Building and measuring this is itself a
  lab for `cml/tf-idf-and-bm25` and `llm/retrieval-augmented-generation-pipeline`.

### 4.7 API & CLI

**CLI first** (ADR-0003). The supported commands below run with `uv run --frozen dogma …`;
`scripts/grade.py` is the earlier standalone practice script.

```text
dogma today [--energy low]        # last transition note, then today's next 45-min session + first action (I-10)
dogma open <problem> [--fresh]    # copy starter → workspace/<problem>.py|.sql; print paths
                                   # --fresh backs up the current file before restoring the starter
dogma run <problem>               # public tests only
dogma submit <problem>            # public + hidden → GradeReport, records the attempt
                                   # (only for your workspace file; --submission FILE runs are
                                   #  not recorded unless --record — authors mutation-test this way)
dogma path [--week N]             # path topics and content status
dogma progress [--week N]         # mastery, attempts and best result per path week
dogma warp [--to-week N | --ahead N | --back N | --original] [--dry-run] [--yes]
                                   # time-warp (I-30a): no option = this week becomes the furthest exercise's week
dogma course reset [--yes]        # backup, clear progress, set workspace aside, week 1 this week; type "reset"
dogma hint <problem> [--level N] [--yes]  # tutor hint ladder L1–L3; L4 walkthrough locked until 2 failed submits
dogma debug <problem>             # names the first wrong concept/line of the latest failing attempt
dogma corpus sync [--no-convert] [--ocr auto] | corpus search QUERY [--topic] [--limit]  # I-11
dogma review [--limit N]          # FSRS card review: reveal, rate 1–4, reschedule (I-09)
dogma mock start [--case ID] | reveal "<question>" | score [--scores-file F] | debrief | abandon | status
                                   # mock interview (I-18c): one open mock at a time; scores stored as answers
dogma close [--stopped-where --next-step --blocker --answer]  # transition note; marks the Learn topic
                                   # introduced and enrols its cards; idempotent; never marks solved
dogma content new|verify|coverage
dogma catalog build              # validate and write var/catalog.json
dogma backup | db upgrade         # JSON export to var/backups/ · migrate the progress DB
dogma serve [--port 8765]         # local browser app (when built) and HTTP API
```

**HTTP API** (`/api/v1`, FastAPI, OpenAPI at `/api/v1/openapi.json` is the contract; I-14): `GET /health`,
`GET /catalog`, `GET /topics/{id}`, `GET /problems/{id}` (never solution/test files), `GET|PUT /workspace/{id}`
(the same file the CLI grades; 256 KiB limit), `POST /attempts` {problem_id, mode} (synchronous, same recording
rule as the CLI), `GET /attempts/{id}`, `GET /today?energy=`, `POST /sessions/{id}/close`, `GET /reviews/due`,
`POST /reviews/{card_id}` (expected_due + content hash; 409 when stale), `GET /progress?week=`, `GET /search`,
`POST /tutor/hint`, `POST /tutor/debug` (JSON), `POST /tutor/hint/stream` and `POST /tutor/debug/stream`
(SSE, I-23: `reviewed` first, then `nudge`/`answer` deltas, then one terminal `done`/`unavailable`/`error`. The UI
replaces streamed text with the guarded text in `done`), and `GET /coverage?tier=&track=` (authored-vs-syllabus overlay).
Time-warp (I-30b): `GET /calendar` (offset, this week, anchor week and problem, course dates, weeks, last move),
`POST /calendar/warp` {kind: sync|to_week|ahead|back|original, week?, weeks?, dry_run} → {applied, changed,
before, after, anchor} (a dry run never writes; a real move with recording off is 409), and `POST /course/reset`
{confirm: "reset"} (422 without the word). `GET /progress` reads the shifted calendar.
Timed re-solves (I-13b): `POST /resolves` {problem_id} starts a window (201 with `window` and `previous_file_kept`;
the same refusals as `dogma resolve`), `GET /resolves/current` answers 200 with the open window (`id`, `problem_id`,
`status`, `started_at`, `budget_s`, `deadline`, `hints_used`) or the JSON value `null` when none is open, and
`POST /resolves/current/abandon` closes it without penalty. A submit's `POST /attempts` response adds `timed_resolve`
(the verdict, time used, hints and the next due item, or null); a run never closes a window. Tutor hint and debug
routes, JSON and SSE, count a hint against an open window on that problem without a prompt. `GET /progress` adds
`readiness` (the next three due re-solves and the last five timed results). Errors are
`application/problem+json` with the domain code. `dogma serve [--host 127.0.0.1] [--port 8765] [--allow-origin]`
binds loopback only, checks Host and Origin (DNS-rebinding guard), and uses explicit CORS origins. `/docs` is
disabled (CDN assets) so the app works offline.
Interview answers and mocks (I-18e, the same services as `dogma answer` and `dogma mock`):
`GET /topics/{id}/interview` (question titles, mastery and the topic-answer rubric with its anchors; the
60-second answers are never sent), `POST /topics/{id}/answers` {question, text | spoken, scores, notes}
(201, with `before`, `after`, `promoted` and `reason`), `GET /rubrics/{topic-answer|mock}`, `POST /mocks`
{case_id?} (201 plan), `GET /mocks/current` (always 200: the `open` plan or null, and `last_scored`),
`POST /mocks/current/reveal` {question}, `POST /mocks/current/score` {scores} (201 report), `POST
/mocks/current/abandon`, and `GET /mocks/{id}/debrief` for any scored mock. `GET /progress` readiness adds
`interview_ready` ({ready, total, weeks[]}), a count of interview-ready topics. Scores are self-given and
never change mastery except through the interview-ready rule.

Attempt responses use a learner-facing report: public test messages remain complete, while
hidden failures use the shared filtered summary also shown by the CLI. Assertion wrappers are
removed before filtering value-bearing lines and Hypothesis examples.

When `web/dist/index.html` exists, the API process serves the local build and falls back to its
index for browser routes. API misses remain structured 404 responses. Every served path,
including the index, must resolve inside the build directory. The served page may call its own
loopback origin; other origins still pass through the existing Host/Origin guard.

### 4.8 Web app (Phase 2)

React + Vite + TypeScript SPA (ADR-0005), served by FastAPI in production mode. Works fully
offline (no CDNs at runtime).

**Visual design:** Memphis Study Club uses a neutral off-white canvas, navy text/actions,
coral and violet decoration, outlined cards and small offset shadows. Today and Start here
use hand-authored SVG figures through the optional illustrated `PageHeading`. Today passes its
current session's `topic_id`; exercise headers use the route's topic ID for a compact scene.
`web/src/components/illustrations/` owns the drawings and `lib/topicIllustration.ts` explicitly
maps authored topics to subject motifs, with a general study scene for unknown or missing IDs.
The artwork is decorative, static, and contains no raster assets or external requests.
Lesson/editor panels remain white and quiet. Decorative colors are separate from the existing
blue/input, orange/pending, green/success and red/error semantics. Normal text targets 4.5:1
contrast; focus/control boundaries target 3:1. Palette decisions and the selected direction
are recorded in `docs/superpowers/specs/2026-11-05-memphis-theme-design.md`.
OpenDyslexic Bold is the interface font; Inter is the Markdown reading font and JetBrains Mono
keeps code aligned. All are bundled locally, including Inter's italic face. Four
OpenDyslexic WOFF2 faces and their upstream license are bundled in `web/public/fonts/opendyslexic/`
with a pinned revision and checksums. `fonts.css` maps all weights to the bold and bold-italic
faces for readability, and the base text weight is bold; `tokens.css` owns the
palette and typography. The learner explicitly prefers light, so OS dark mode does not override
the theme. Reduced-motion support, visible focus, labeled status colors and keyboard shortcuts
remain active. Mobile session tables reflow into readable steps; code and wide examples scroll
inside their own panels.

Vite emits bundled assets as local files (`assetsInlineLimit: 0`), including small KaTeX and
JetBrains font subsets, to satisfy the API's `font-src 'self'` policy without blocked data URLs.

**Markdown:** `react-markdown` uses `remark-gfm` for tables and other GFM constructs, and
`remark-math` with `rehype-katex` for locally rendered formulas. `remarkDisplayMath` promotes
paragraphs containing only `$$equation$$` to display math, supporting the lessons' one-line
syntax without changing inline math or code. Tables retain semantic headers inside a labeled,
keyboard-scrollable region. Wide tables and display equations scroll within the reading panel.
Raw HTML remains disabled and link targets still pass the renderer's URL checks.

I-15 implements the shell and Problem page in `web/`: CodeMirror, sanitized Markdown/KaTeX,
streamed hints, learner-safe results and an autosaved editor. Run/Submit wait for the current
buffer to save. Failed departure saves retain the unsaved buffer in session memory for recovery;
unload prompts protect active and failed saves. Saved workspace files and SQLite remain the
durable state. OpenAPI generates the client types; `pnpm gen:api --check` detects stale artifacts.
`make check` also checks and builds the browser app. The current checkout also implements
Today/Review (I-16) and Map/Progress (I-17); their acceptance status remains in
[ROADMAP.md](ROADMAP.md). The root opens `/welcome` on a first visit in a browser and `/today` after that; unknown browser routes show a
Not Found page.

| Route | Contents |
|---|---|
| `/welcome` | Start here: what dogma is, the session shape, where things live, a first-problem tutorial and a glossary. Opening it marks it seen in browser storage, and it stays in the nav |
| `/today` | Last transition note, session blocks, first action, energy mode and session-close form. A planned timed re-solve is a "Start timed re-solve" action that opens its Problem page. When the calendar is two or more weeks past the furthest exercise, one neutral line links to the Progress page's Calendar section (I-30b) |
| `/problem` | Published problem index |
| `/problem/:track/:topic/:problem` | `?tour=1` adds a dismissible guide strip. Statement / hints / lesson / answer (the topic's interview question: write it or mark it spoken, then self-score the four dimensions with their anchors shown; the saved result says what it did to mastery), CodeMirror editor, Run (⌘↵), Submit (⌘⇧↵), streamed hints and results. While a timed re-solve of this problem is open: budget, a calm countdown, the rules, Abandon (with confirmation), a hint confirmation, and the verdict with the next due date after a submit |
| `/review` | Due card → reveal → Again / Hard / Good / Easy, with rating keys 1–4 |
| `/mock` | Timed mock (I-18e): start, the case brief and planned part timings (a plan, never a red deadline), a question box that calls the interviewer's reveal, the 30-item scoring form grouped by part with the anchors shown, and the debrief (part totals, band, strongest part, area to improve, the case's debrief). Today's Interview session links here. Abandon asks first |
| `/map` | Syllabus coverage, authored content status and mastery, with tier/track filters |
| `/progress` | Core-path weeks, solved topics, attempts and best results, and a Readiness section: the next due timed re-solves and the last timed results. A Calendar section (I-30b): this week, the furthest exercise's week, and moves (match my exercises, go to week N, back to the original dates), each previewed before it applies. A Start over section resets the course after a typed `reset`. **No streaks** (ADR-0013) |

Lessons render within the Problem page; a separate interactive Lesson page remains planned. The
timed Interview runs on `/mock`. The Progress page shows recorded results, the Readiness section, and
an Interview-ready section with counts per study week (I-18e). Retention metrics depend on later
learning-engine increments.

### 4.9 Storage

SQLite at `var/dogma.db` (WAL mode). SQLAlchemy 2.0 models, Alembic migrations, forward-only.

| Table | Key columns |
|---|---|
| `topic_state` | topic_id, mastery (introduced\|solved\|consolidating\|retained\|interview-ready), introduced_at, solved_at, retained_at, ready_at |
| `attempt` | id (= GradeReport attempt_id), problem_id, problem_version, mode (run\|submit; timed/mock later), status, public_passed/total, hidden_passed/total, hints_used, run_dir (repo-relative), started_at, submitted_at, duration_ms |
| `resolve_window` (I-13a) | id, problem_id, problem_version, topic_id, started_at, budget_s, hints_used, status (open, passed, partial, failed, abandoned), attempt_id (the deciding submit), closed_at · at most one open row (partial unique index) |
| `study_session` | id, plan_date (local), slot, kind (learn\|consolidate\|minimum\|interview), energy, status (open\|closed), topic_id, must_do, first_action, plan_notes, created_at, closed_at, stopped_where, next_step, blocker, self_answer, timed_problem_id (nullable) — UNIQUE(plan_date, slot) (I-10, I-26) |
| `session_block` | id, session_id, position, name, minutes, detail (I-10) |
| `card` / `card_review` (I-09) | stable card id, topic, content hash + Q/A snapshot, active, FSRS state/step/stability/difficulty, due, last_review · per-rating event with due before/after and content snapshot |
| `answer` (I-18a) | id, topic_id (NULL for mocks), kind (interview\|close\|mock), prompt_ref (e.g. `interview.md#Q1`, or a case id), text (NULL when answered aloud; for a mock, the item notes), rubric_id, rubric_version, scores (json: criterion id → 1–4), notes, created_at, mock_session_id (I-18c, mock scores only), model_score (I-18d, nullable json: advisory model, prompt version, rubric id and version, dropped count, and per-criterion score with quote; never read as a self-score) |
| `mock_session` (I-18c) | id, started_at, ended_at, case_id, coding_problem_id, theory_topic_id, theory_question, status (open\|scored\|abandoned; one open at a time), part_minutes (json) |
| `calendar_warp` (I-30a) | id (autoincrement; the newest row is in force), created_at, kind (sync\|to_week\|ahead\|back\|original\|course_reset), offset_weeks, previous_offset_weeks, week_before, week_after · append-only; a course reset keeps it |
| `tutor_message` | id, attempt_id?, topic_id, mode, role, content, citations (json), model, prompt_version, ts |

Attempts are keyed by `(problem_id, problem_version)`, so editing a problem never corrupts
history. Attempts, topic states, cards, card reviews, study sessions, session blocks and timed
re-solve windows now exist (I-08 through I-10, I-13a), and interview answers with their rubric
scores (I-18a). Mock sessions and mock answers were added by migration in I-18c; the advisory model score column
on `answer` came in I-18d (one nullable column, no table, no backfill). Tutor messages come later.
Rubric files live in `curriculum/rubrics/` as versioned YAML, not in the database: each stored score
keeps the rubric id and version it was given under. The window table holds no schedule: the next due time is
computed from closed windows each time it is read.
`dogma backup` first saves an existing SQLite database, including committed WAL rows, as an
atomic `.sqlite3` snapshot under `var/backups/`, before opening a session can migrate it. A
failed snapshot aborts before migration. It then exports every current-schema table as JSON
(`format`, `format_version`, `exported_at`, `schema_revision`, `tables`). A new database has no
pre-existing state to snapshot and produces only JSON. Copy backups to the External SSD too.
Recording can be switched off
globally with `DOGMA_RECORD_ATTEMPTS=false`.

Since I-26, session plans persist `timed_problem_id`; Today resolves the action from that id
and reads whether its window is actually open. Migration `b8c9d0e1f2a3` backfills only the known
legacy timed-action templates, preserving existing instructions. Runtime code does not parse
instruction text to identify a timed re-solve.

An open session resumes across local midnight and keeps its original study date when closed.
Closing, introducing its topic and enrolling review cards share one transaction; a failed write
leaves the session open for retry. A review queue entry carries both its due time and content
hash. Ratings must match both, so edits cannot attribute a rating to unseen content.

## 5. Pedagogy → feature map

The general requirements in [Learning Design Principles](LEARNER_PROFILE.md) map to concrete
features. These requirements describe product behavior without profiling an individual.

| Principle | Feature |
|---|---|
| Explicit > implicit; teach systems, not tricks | Every lesson opens with "The system" (rules, subgroups, decision tree) before examples |
| Bounded uncertainty | Today page shows the whole session shape and time up front. Each problem states "done looks like…". Lessons include a *Signal vs. noise* box (what matters for interviews, what doesn't) |
| Predictable structure, flexible content | Five fixed session blocks. Fixed lesson section order. Fixed hint ladder |
| Retrieval before re-teaching | Retrieve block opens every session. Predict question comes before every lesson |
| Spacing & interleaving | FSRS cards. Timed re-solves with 7/21/60-day gaps. One interleaved old problem per session |
| Generation effect | Predict block. Hints are rationed (the ladder unlocks gradually) |
| Name the error type | Grader messages and tutor `debug` say *what kind* of mistake it is |
| Practise recall under time constraints | Timed re-solves. Weekly mocks. `interview-ready` requires a timed, hint-free solve |
| Never cold-start on something hard | Retrieve block uses familiar material first |
| Make starting straightforward | Pre-chosen first action. 10-minute start counts as a win. One must-do |
| Never start with archaeology | Transition note written at Close, shown first next time |
| Support shorter practice sessions | Low-energy mode. `min_viable` path per topic. Missed days absorbed, no debt counters, no streaks |
| Errors are data; co-pilot tone | Tutor tone contract. Mastery never resets to zero |
| Visual support | Semantic colour palette, diagrams, interactive widgets in lessons |
| Connect ideas to applications | Interview angles use synthetic cases and public examples. Inference labs use locally configured models |

## 6. Key flows

**Solve a problem**: `open` → edit → `run` (public) → `submit` (public + hidden) → runner →
harness → GradeReport → attempt saved when recording is enabled. A passing core submit marks
the topic `solved`. Hints and debug are explicit learner requests; tutor probes are planned.

**Daily session**: `today` → planner reads state → plan saved to `study_session` and
`session_block` → learner works through it → `close` saves the answer and transition note.
Closing a Learn session also introduces its topic and enrols cards in one transaction.

**Timed re-solve**: `dogma resolve <problem>` opens a window and restores the starter. The next
`dogma submit` of that problem closes it in the same transaction as the attempt, sets its verdict,
re-evaluates the topic's mastery and reports the next due time. `run` and `--submission` grading
never close a window, and `DOGMA_RECORD_ATTEMPTS=false` never opens or closes one. Consolidate
plans due re-solves, and `dogma progress` shows the next three due and the last five results.

**Author a topic** (§7): template → draft → `content verify` → `verified` → review → `reviewed`.

## 7. Content production pipeline

Content is the product and the main risk. The pipeline:

1. **Tier the syllabus**: ✅ done 2026-10-23 (`tier:` on all 538 topics + `curriculum/paths/core.yaml`).
2. **Draft** (agent, batched by track in path order): `dogma content new <id>` scaffolds from
   templates. The agent writes lesson, problems, tests, cards and interview notes from
   `sources/` + corpus retrieval, with citations.
3. **Verify** (automatic, `dogma content verify`): schema valid; solution passes public and
   hidden tests twice with identical outcomes; starter runs and fails ≥1 test; at least three
   public and five hidden tests; static checks and lint pass. The full `make check` also runs
   the privacy audit. Authors set `status: verified` after the checks pass; verification does
   not rewrite metadata. Oracle coverage is required by the content review standards.
4. **Review** (decided: a second agent, not the learner): an independent agent pass with the
   checklist in ENGINEERING_STANDARDS §5.6 → `reviewed`, or a list of fixes for the author agent. The UI shows a *verified, not yet reviewed* badge, so you always know how much to
   trust a page.
5. **Runway rule:** content stays ≥1 week ahead of the learner. While the runway is shorter,
   platform feature work pauses and authoring takes priority.

Throughput target: one week of the core path (about 13 topics) per content wave of 3 parallel
author agents plus 1 reviewer. **Runway rule:** week N+1 is verified before week N starts.

## 8. Repository layout

```text
dogma/
├── README.md  CLAUDE.md  pyproject.toml  uv.lock  Makefile  .python-version  .env.example
├── docs/        DEVELOPMENT  ARCHITECTURE  ENGINEERING_STANDARDS  CONTENT_SPEC  ROADMAP  IMPORT_PLAN  LEARNER_PROFILE  decisions/
├── curriculum/  syllabus.yaml  paths/core.yaml
├── content/     <track>/<topic>/…                          # the curriculum (committed)
├── sources/     manifest.yaml  manifest.lock.json  <id>/…  # imported seed material (committed, scrubbed)
├── templates/   topic + problem scaffolds for `content new`
├── scripts/     grade.py (legacy loop)  import_sources.py  audit_privacy.py  check_commit_files.py  check_content_types.py
├── src/
│   ├── dogma/
│   │   ├── core/      config, errors (codes), fsio (atomic writes)
│   │   ├── content/   schemas, loader, catalog, verify, coverage
│   │   ├── runner/    sandbox, run dirs, watchdogs
│   │   ├── grading/   GradeReport, static checks
│   │   ├── learning/  mastery, FSRS adapter, planner, sessions
│   │   ├── tutor/     gateway, providers/, modes, prompts/, citations
│   │   ├── corpus/    docmd adapter, chunker, FTS index, search
│   │   ├── db/        SQLAlchemy models, Alembic migrations
│   │   ├── api/       FastAPI app
│   │   └── cli/       Typer app
│   ├── dogma_data/   # test-only data layer: real sets, HF snapshots, synth; numpy at import
│   └── dogma_harness/   # runs inside the sandbox; stdlib + pytest; never imports dogma
├── tools/docmd/ vendored PDF→Markdown converter (own uv project, MIT)
├── web/         React + Vite + TS (from I-15)
├── tests/       unit/ integration/ e2e/ fixtures/
├── corpus/      raw/ md/ index.db                         # gitignored (third-party works)
└── private/  var/  workspace/                             # gitignored
```

**Shared domain helpers** (I-23): `content/resolve.py` (ids and workspace paths), `content/coverage_scope.py`,
`learning/path.py` (calendar and topic status), `runner/attempt.py` (the grade seam), `api/streaming.py` (SSE wiring).

**Dependency rules** (enforced with import-linter): `core` ← everything; `content` ← `grading`,
`learning`, `tutor`; `runner` ← `grading` only through the GradeReport schema; `api`/`cli` are the
only layers that wire components together; `dogma_harness` imports nothing from `dogma`; `web` talks
only to `/api/v1`.

**Test data** (`src/dogma_data/`) is test-only. At import time it loads only numpy; other libraries
load inside the functions that need them. It never imports `dogma` or `dogma_harness`, and neither
imports it. The submission static check bans it. The opt-in `hf` dependency group is used only by
`scripts/sync_hf_datasets.py`, which vendors text snapshots; tests never install or call it.

## 9. Decision log

New decisions get `docs/decisions/NNNN-title.md` (template in ENGINEERING_STANDARDS §12). The
founding decisions are recorded here:

| ADR | Decision | Why / alternatives rejected |
|---|---|---|
| 0001 | Local-first, single user, no auth | Local deployment avoids account, multi-tenancy and hosting infrastructure |
| 0002 | Curriculum-as-code (YAML + Markdown + Python in git) compiled into a catalog | Diffable, testable, agent-authorable. A CMS or DB-backed content loses verification and history |
| 0003 | CLI first, web second | Study starts around day 3 instead of week 3. The CLI + VS Code is already a good loop |
| 0004 | Python 3.12 + uv, FastAPI, SQLite + SQLAlchemy/Alembic, Typer, Pydantic v2 | A locked, compatible toolchain for the platform and its numerical runtimes |
| 0005 | React + Vite + TS SPA, CodeMirror 6, KaTeX, TanStack Query, TS types generated from OpenAPI | An editor-heavy interactive UI. Generated types prevent front/back drift. HTMX was rejected for the editor and result interactivity |
| 0006 | Run code locally in a subprocess sandbox, not Pyodide | Needs torch, JAX, DuckDB and real performance. Single trusted user |
| 0007 | Grading by tests + oracles. The LLM never decides correctness | LLM judgments are noisy. Oracles make even LLM-authored problems trustworthy |
| 0008 | Local LLM through a gateway. Default `qwen3.8-27b-q6`. External providers opt-in per request | Privacy, offline use, no cost. One resident model avoids 30–200 s router swaps |
| 0009 | BM25 (SQLite FTS5) first. Embeddings only if measured recall needs them | No embedding model in the vault. BM25 is strong on technical text. Measure before adding complexity |
| 0010 | FSRS for card scheduling, target retention 0.90 | State of the art, adapts to your own data. SM-2 was rejected |
| 0011 | Private/runtime material gitignored + privacy audit gate. Source handling superseded by ADR-0014: scrubbed `sources/` is committed | Publishable by construction. Personal and licensed material can't leak |
| 0012 | GPU tracks via CPU simulation (numba CUDA simulator, numpy Triton shim). Real GPU optional later | Apple Silicon has no CUDA, and Triton has no macOS wheels. The programming model is what interviews test |
| 0013 | No streaks, leaderboards or debt counters. Progress = retention + readiness | Streaks punish missed days and reward volume over durability |
| 0014 | Self-contained repo: imported material committed (scrubbed), tools vendored (`tools/docmd`), third-party PDFs copied into gitignored `corpus/raw/`. Only the AI runtime stays external | Other projects on this machine are temporary. The platform must outlive them |
| 0015 | Harness contract is pytest-based: tests import `submission` from a fresh run dir. One contract for Python and SQL | Authors verify problems with plain pytest before the platform exists (`scripts/grade.py`). No custom test DSL to learn |
| 0016 | No PII or employer/client data anywhere in the repo. Personal context only in gitignored `private/`. `make audit` gate | Publishable by construction. Replaces the "personal narratives" part of ADR-0011 |
| 0019 | Personal course calendar: a whole-week offset in SQLite over the authored path (time-warp); review due dates stay real | Breaks never leave the calendar ahead of the work, and authored content never holds learner state |
| [0018](decisions/0018-context-independent-publication.md) | General requirements and examples; working-tree/build privacy checks, staged-blob checks at commit, and history-free source publication | Keep individual context out of product artifacts while preserving required upstream attribution and local development history |

## 10. Roadmap

The build is split into **small increments, each sized for one coding session**, tracked in
[ROADMAP.md](ROADMAP.md) with status, owned paths, acceptance criteria and a handoff log, so work
can stop and resume across sessions. Milestones:

| Milestone | What it gives the learner | Increments |
|---|---|---|
| **M0: Study can start** | Verified week-1 content + `scripts/grade.py` practice loop | I-00 … I-04 |
| **M1: Real CLI loop** | `dogma open/run/submit`, sandboxed runner, content gate | I-05, I-06 |
| **M2: Memory** | Progress DB, FSRS card reviews, `dogma today` planner, transition notes | I-08 … I-10 |
| **M3: Tutor** | Corpus search + local-model hints and debugging | I-11, I-12 |
| **M4: Browser** | API + web app: Today, Problem, Review, Map | I-14 … I-17 |
| **M5: Interview layer** | Timed re-solves, mocks, readiness dashboard, sys track | I-13, I-18 |

Content waves (one per path week) run in parallel with all of the above.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Building the platform eats study time | CLI first. The runway rule pauses features when content falls behind. Phase exit criteria are about *studying*, not features |
| Errors in LLM-authored content | Oracle tests, determinism checks, citations, verified/reviewed badges |
| Local model too slow for interactive hints (27B dense) | Streaming, small contexts, short hint formats. Fallback `qwen3.6-35b-a3b-q4` (MoE, faster decode) via config, at the cost of a router swap |
| External SSD not mounted | Tutor degrades gracefully. Everything else is local to the repo |
| `sandbox-exec` deprecated on macOS | It's defence in depth. Timeouts, process groups and the RSS watchdog are the real limits |
| numba CUDA simulator / Triton shim fidelity | Phase-4 spike before authoring gpu content. Those topics are tier B/C anyway |
| Uneven energy across weeks | Low-energy mode, `min_viable` paths, absorbed missed days, no streaks |

## 12. Decisions on the open questions (2026-10-23)

- **Q1 → a structured core course, defaulting to 3 × 45 min per study day.** Tier A = 164 topics on a 13-week path
  plus an interview week (§2).
- **Q2 → balanced** DS and AI engineering.
- **Q3 → a second agent reviews content** (§7).
- **Q4 → publishable by construction**: self-contained repo, no PII (ADR-0014, ADR-0016).
