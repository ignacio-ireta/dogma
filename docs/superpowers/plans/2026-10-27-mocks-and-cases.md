# Mock Interviews, Rubric Scoring and the `sys` Track (I-18)

**Goal:** the interview layer of M5. The learner can (1) answer a topic's interview question and
score it against a fixed rubric, which with `retained` earns `interview-ready`; (2) run a timed
mock (coding + ML theory + a case) scored with the imported 30-item mock rubric; (3) practise
system-design cases from a local `sys` track. Weak rubric dimensions feed the planner.

**Spec:** `docs/ARCHITECTURE.md` §2 (`sys` track), §4.1 (content types `interview`, `case`), §4.4
(mastery chain, Interview session, weekly mock), §4.5 (`score`, `interviewer` modes), §4.9
(`answer` table); ADR-0007 (the LLM never decides correctness), ADR-0013 (no streaks/debt);
`docs/LEARNER_PROFILE.md`. Imported sources: `sources/interview-prep/mock-interviews/`
(`SCORING_RUBRIC.md`, packs A–F), `sources/interview-prep/drills/case-scenarios/`,
`sources/interview-prep/frameworks/RASS.md`.

## Decisions

- **Rubrics are data**, versioned YAML under `curriculum/rubrics/`, loaded and validated by
  `dogma.content` (Pydantic, `extra="forbid"`):
  - `mock-v1.yaml`: the imported rubric. Six parts (coding, reframe, ask, structure, synthesize,
    communication), 30 items with stable ids (`coding.narration`, …), scale 1–4 with the four
    anchor descriptions, part maxima, interpretation bands. Text paraphrased, not copied verbatim.
  - `topic-answer-v1.yaml`: four dimensions for a topic's spoken/written interview answer:
    `accuracy`, `structure` (answer shape from the lesson's Interview angle), `trade-offs`
    (caveats, when it breaks), `communication` (concise, plain language). Scale 1–4, anchors.
  A rubric version is stored with every score, so editing a rubric never rewrites history.
- **Self-scoring is the source of truth.** The learner scores; a model score (I-18d) is advisory,
  shown next to it with evidence quotes, and never changes mastery on its own (ADR-0007).
- **`answer` table** (ARCHITECTURE §4.9): `id`, `topic_id` (nullable for mocks), `kind`
  (`interview|close|mock`), `prompt_ref` (e.g. `interview.md#Q1` or a case id), `text`,
  `rubric_id`, `rubric_version`, `scores` (JSON: item id → 1–4), `notes`, `created_at`. Plus a
  `mock_session` table: `id`, `started_at`, `ended_at`, `case_id`, `coding_problem_id`,
  `theory_topic_id`, `status` (`open|scored|abandoned`), `part_minutes` (JSON). One
  forward-only migration.
- **`interview-ready`**: a `retained` topic becomes `interview-ready` when an `interview` answer
  for it is self-scored ≥ 3 on every `topic-answer-v1` dimension (set `ready_at`). A failed timed
  re-solve or a card lapse still drops it to `consolidating` (existing rules); a later qualifying
  answer after re-retention promotes it again.
- **`sys` track** (`origin: local`) added to `curriculum/syllabus.yaml` with new stable ids, tier
  B (A stays exactly the core path), e.g. `sys/rass-case-framework`, `sys/churn-prediction-case`,
  `sys/fraud-detection-case`, `sys/predictive-maintenance-case`, `sys/recommendation-system-case`,
  `sys/logistics-optimization-case`, `sys/rag-system-design-case`. New content type **case**: a
  topic package whose `topic.yaml` has `kind: case` and which holds `brief.md` (candidate-facing
  setup), `reveals.yaml` (list of `{when_asked, answer}` the interviewer releases only on a good
  question), `twist.md`, `debrief.md` (a strong answer's outline per rubric part, hidden until
  scored), optional `exhibits/` (PNG/CSV), plus the usual `lesson.md`/`cards.yaml`. No auto-graded
  problems in a case package. The validator, loader, catalog and coverage learn the new kind.
  Seed cases come from the imported scenarios and packs, scrubbed and paraphrased (no client
  names that look real; keep the fictional companies only if the audit stays clean).
- **Mock sessions** (`dogma mock`): `start [--case ID]` picks a case (weakest recent rubric part
  first, then least recently used), a verified core coding problem the learner has solved, and a
  theory question from a retained topic's `interview.md`; prints the plan with part timings (coding
  10–15', reframe 2', ask 4', structure 12', synthesize 3'); `reveal "<question>"` prints a
  matching reveal only when the question's keywords match (deterministic keyword match; the LLM
  interviewer is I-18d); `score` walks the 30 items (1–4 each, optional note) and stores them;
  `debrief` shows part totals, the band, the top strength and top area to improve, and the
  case's `debrief.md`. `abandon` closes without scores.
- **Planner:** from path week 3, one Consolidate per week becomes an Interview session (Mock 40' →
  Debrief 5'); every session in week 14 is an Interview session. Low-energy days never get one.
  The weakest rubric part over the last three scored mocks is named in the plan notes. No
  scores trend lines or streaks.

## Split

| Part | Scope | Depends on |
|---|---|---|
| **I-18a** | Rubric YAML + schema/loader, `answer` table + migration, `dogma answer <topic>` (show interview.md Q, record a written answer or "spoken" marker, self-score on topic-answer-v1), `interview-ready` rule, readiness shows interview-ready counts | I-13a |
| **I-18b** | `sys` track in the syllabus, `case` content kind (spec, schema, validation, catalog, coverage, `dogma content new` template), seed 3 cases (`sys/rass-case-framework` as a lesson-style topic plus churn and fraud cases) | — |
| **I-18c** | `mock_session` table, `dogma mock start/reveal/score/debrief/abandon`, planner Interview sessions and weak-part notes | I-18a, I-18b |
| **I-18d** | Tutor `score` (advisory, evidence quotes, versioned prompt + golden tests) and `interviewer` (releases reveals on good questions) through the gateway | I-18a, I-18c |
| **I-18e** | API + browser: answer/score forms, mock flow, readiness shows interview-ready | I-18a, I-18c |

## Ownership

- I-18a: `curriculum/rubrics/**`, `src/dogma/content/rubrics.py` (+ schema), `src/dogma/db/`
  (answer model + migration + backup), `src/dogma/learning/` (answers service, mastery rule),
  `src/dogma/cli/answer_cmd.py` + one registration line, `cli/progress.py`, tests, ARCHITECTURE
  §4.4/§4.9 wording.
- I-18b: `curriculum/syllabus.yaml` (append the `sys` track only), `src/dogma/content/**` (case
  kind), `templates/case/**`, `docs/CONTENT_SPEC.md` (new §9 Case packages), `content/sys/**`,
  tests under `tests/unit/content/**` and fixtures.

## Verification

Each part: `uv run --frozen pytest -q`, relevant `-m e2e`, ruff, mypy, lint-imports; I-18b also
`uv run --frozen --group torch dogma content verify --all` and `make audit`. Never touch the
real learner DB; scratch Settings only. Coordinator runs the full gate per merge.
