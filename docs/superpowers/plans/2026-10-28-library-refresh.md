# Library Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve existing content using newly acquired primary sources, then close selected optional content gaps without extending the core schedule.

**Architecture:** Add an explicit, checksum-verified source refresh alongside the original snapshot. Revise topic packages in bounded batches and preserve ids, existing citations and learner progress. Keep optional extensions outside the core path.

**Tech Stack:** Existing Python 3.12/uv importer, YAML curriculum, Markdown content, pytest/Hypothesis/oracles, docmd and SQLite FTS5. No new dependencies.

**Spec:** [Library refresh design](../specs/2026-10-28-library-refresh-design.md), especially the exact topic ownership map; [frozen 108-file inventory](../../source-refresh-2026-10-28.json).

## Global Constraints

- No runtime/build dependency outside this repository.
- Preserve the 164-topic, 13-week core path and interview week.
- No content id renumbering; no promotion to tier A or core-path scheduling changes.
- Preserve scrubbed source files, lock provenance and learner history.
- Keep problem ids stable; bump problem versions when tests, tolerances or contracts change, as CONTENT_SPEC requires.
- Copyrighted PDFs and conversions stay gitignored.
- No new dependencies. Use `uv run --frozen`.
- A status is `reviewed` only after a second content reviewer completes the review.

## Review Focus

- Moved upstream paths must not delete old imported files or invalidate existing `library:` citations (Task 1).
- A source whose bytes differ from the inventory must stop import before any overwrite (Task 1).
- A note containing a local vault link or sensitive material must be scrubbed rather than exposed by the tutor (Task 1).
- A lesson may overstate guarantees on singular matrices, dependent samples or shifted test data (Tasks 2–4).
- Optional topics and cases must remain discoverable without being inserted into the learner's core plan (Tasks 5–6).

---

### Task 1 / I-25a: Import only the reviewed source refresh

**Files:** `scripts/import_sources.py`, `sources/manifest.yaml`, `sources/manifest.lock.json`,
`tests/unit/scripts/test_import_sources.py`, `docs/IMPORT_PLAN.md`; new notes under
`sources/library-refresh/`; PDFs under gitignored `corpus/raw/library-refresh/`.

**Interfaces:** consumes the inventory's `items[].path` and `sha256`; produces stable
`source: library-refresh/<Library-relative note path>` citations and converted corpus source ids.

- [ ] Add importer support for a source-id selector, `--source-id library-refresh`, so a refresh
  does not require every original import root. Add optional corpus `include` paths; use the
  existing glob behavior when absent. Reject missing or checksum-mismatched allowlisted files
  before copying. Unit fixtures must show: mismatched bytes → no copies; selector → no unrelated
  roots read; rerun → no writes; a pre-existing scrubbed copy → preserved without `--force`.
- [ ] Extend the manifest with a `library-refresh` snapshot rooted at logical `library`, using
  exact import rules for the inventory's 36 notes, and a `library-refresh` corpus entry rooted
  at `library`, with exact `include` paths for its 72 PDFs. Resolve the authorized Library root
  only in the gitignored local roots file; never commit an absolute origin path. Do not change
  existing `library`/`library-papers` rules or lock entries.
- [ ] Add a checked checksum input option `--inventory docs/source-refresh-2026-10-28.json`
  with signature `validate_inventory(root: Path, inventory: Path) -> None`; all selected files
  must match before the importer writes anything. Missing/mismatched files exit nonzero with
  relative paths only. Existing import invocations retain their behavior.
- [ ] Run the failing importer fixtures, implement the minimal selector/include/checksum
  changes, then run `uv run --frozen pytest tests/unit/scripts/test_import_sources.py -q`.
  Expected: all importer tests pass, including preservation of existing scrubbed destinations.
- [ ] Run `uv run --frozen python scripts/import_sources.py --source-id library-refresh
  --inventory docs/source-refresh-2026-10-28.json`. Review copied notes and links; remove or
  rewrite references to unimported vault material, personal contexts and source file paths.
  Preserve bibliographic author names. Relock after scrubbing. Check no tracked old source is
  removed and record a disposition for the 10 unselected older/alternate PDFs; they are outside
  this allowlist unless separately reviewed.
- [ ] Run `dogma corpus sync` via frozen uv, inspect conversion failures and selected primary
  passages, and confirm FTS retrieval for `LoRA`, `DPO`, `conformal`, `normal equations` and
  `linear programming`. Record exact indexed source ids before writing `corpus:` references.
  PDFs and conversions remain ignored; notes/lock entries carry origin and final hashes.
- [ ] Verify two identical import runs produce no diff, `make audit` and `make check` pass;
  commit import/tooling separately from subsequent content revisions.

### Task 2 / I-25b: Numerical foundations, autodiff and optimization

**Files:** complete topic packages in the first three I-25b rows of the spec; no syllabus edits.
**Interfaces:** consumes Task 1's stable note/corpus ids; produces source-backed lessons,
cards and interview questions with the existing topic/problem ids.

- [ ] Read the named primary sources at the relevant sections; record locators alongside each
  changed claim. Compare current lessons first and skip redundant additions.
- [ ] Reproduce existing behavior on rank-deficient and nearly-collinear least squares, shared
  autograd subexpressions and nonsmooth L1 shrinkage. Add failing oracle/gradient tests only
  where the current contract or verification misses the behavior. Keep an inverse formula's
  teaching exercise distinct from recommendations about numerically stable solvers.
- [ ] Revise the numerical reasoning, derivative-shape explanations and optimization assumptions.
  Add a small worked counterexample, one retrieval card and one interview follow-up where that
  improves the lesson. Use optional problems for new algorithms; do not lengthen p01 or weeks.
- [ ] Verify only changed topic ids first with `dogma content verify <id>...` via frozen uv
  and the torch group where needed. Expected: reference solutions pass; starters fail usefully;
  gradients match finite differences under documented numerical tolerances.
- [ ] Have a second reviewer check conditioning claims, transposes/JVP/VJP and convexity/step-size
  assumptions. Run `make check`; commit only `content/` changes as a separate content commit.

### Task 3 / I-25c: Statistical assumptions and uncertainty

**Files:** complete topic packages in the two I-25c rows of the spec.
**Interfaces:** consumes Task 1 source ids; preserves existing contracts and conformal rank rule.

- [ ] Verify the primary-source locators and current lessons. Identify precise changes for
  conditioning, likelihood/MAP, confidence/power, bootstrap assumptions, temporal leakage,
  identification and marginal coverage.
- [ ] Add meaningful checks where missing: fixed seeded bootstrap/Monte Carlo oracles, a
  blocked/time-dependent resampling counterexample, finite-sample conformal rank edge cases
  and a covariate-shift example framed as a limitation rather than a promised guarantee.
- [ ] Improve the worked examples, assumptions, cards and interview answers; avoid turning
  population-risk explanations into a required measure-theory detour.
- [ ] Run changed-topic verification and a second review focused on inference versus prediction,
  causality and exchangeability. `make check` must pass before the content-only commit.

### Task 4 / I-25d: Precision, adaptation and primary-paper citations

**Files:** complete topic packages in the two I-25d rows; select the exact existing RAG/MoE/RLHF
packages during source review and list their owned paths in the handoff before editing.
**Interfaces:** consumes Task 1 corpus ids; unsupported citations remain explicitly unsupported.

- [ ] Verify Goldberg's numerical examples, Hu et al.'s LoRA equations and Rafailov et al.'s
  DPO objective directly. Compare existing lessons and tests before editing.
- [ ] Check range-versus-precision examples, LoRA scale/initialization/parameter arithmetic and
  DPO beta/reference-policy claims with independent arithmetic or gradient checks. Use the
  complete Constitutional AI and hierarchical-MoE versions, not the older summary/conference
  copies, when making claims about the full papers.
- [ ] Replace memory-only provenance only when the cited primary passage was inspected. Keep
  Sutton–Barto, PagedAttention and Bradley–Terry acquisition gaps explicit. Update cards and
  interview follow-ups only where meaning changes; do not duplicate the existing lesson.
- [ ] Run changed-topic verification, second review and `make check`; commit content separately.

### Task 5 / I-25e: Finish selected optional topic packages

**Files:** `content/cml/lagrange-multipliers/`, `content/cml/gaussian-mixtures-and-em/`,
`content/llm/document-packing-and-cross-document-masks/`, `content/rl/sarsa-and-expected-sarsa/`.
**Interfaces:** existing syllabus ids and tier-B status; no edits to `curriculum/paths/core.yaml`.

- [ ] Scaffold each existing id with `dogma content new <id>`, then author original lessons,
  cards, interview notes and core exercises from the appropriate primary sources. Acquire or
  explicitly qualify missing sources for packing/SARSA before writing technical claims.
- [ ] Pin constrained-stationary versus optimum distinctions and EM responsibility/likelihood
  arithmetic with independent oracles; pin cross-document causal-mask boundaries and expected
  SARSA's policy-weighted target with hand-computable fixtures and independently coded checks.
- [ ] Verify each topic, confirm tier-A/core-path sets remain unchanged, obtain second review,
  run `make check` and commit each independently useful topic batch.

### Task 6 / I-25f: Complete the optional system-design cases

**Files:** the four exact `sys` case paths listed in the spec; existing case templates and rubric
are references, not files to change. No new track or new mandatory study sessions.
**Interfaces:** existing case schema, stable `sys` ids and `mock-v1` scoring rubric.

- [ ] Author predictive maintenance, recommendations, logistics and RAG cases in separate small
  batches: synthetic brief, earned reveals, exhibits, twist, debrief and trade-off questions.
  Use LP certificates and evaluation-budget/Pareto reasoning where the scenario warrants them.
- [ ] Check ambiguous/repeated reveal questions, unavailable facts, solution leakage in briefs,
  unrealistic metric definitions and the distinction between model quality and decision value.
- [ ] Run case validation and the scratch browser mock flow. Confirm the cases appear in the
  catalog/Map and can be selected by mocks without entering the core topic path. Obtain second
  review and `make check`, then commit content separately.

## Completion and handoff

Update each I-25 sub-increment in `docs/ROADMAP.md` only after its acceptance evidence exists.
Record exact source sections, problem-version changes, independent reviews, commands and remaining
source gaps. This session delivers the audit and this plan; it does not mark the refresh authored
or imported. Execute Task 1 first, then prioritize a batch that serves the learner's current week.
