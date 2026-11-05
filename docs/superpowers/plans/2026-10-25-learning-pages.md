# Learning Pages Implementation Plan (I-16 / I-17)

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Work test-first on behavior and keep the two page groups independent.

**Goal:** Complete the browser's Today, Review, Map and Progress pages using the existing API and UI.

**Architecture:** Reuse the local API as durable truth, TanStack Query for server state,
generated OpenAPI types, existing Markdown/KaTeX, navigation and design tokens. Keep
new page groups in separate modules. Coordinator owns route integration and browser smoke tests.

**Tech Stack:** Existing React/TypeScript/TanStack Query/Vitest; Playwright for a small real-browser smoke suite.

**Spec:** `docs/ARCHITECTURE.md` §4.4/4.7/4.8,
`docs/LEARNER_PROFILE.md`, `docs/superpowers/specs/2026-10-24-learning-sequence-design.md`.

## Global Constraints

- Preserve the I-15 shell, workspace recovery and editor behavior; no redesign or new state store.
- Derive API response/request types from `src/api/schema.d.ts`; do not handwrite response schemas.
- SQLite and workspace files remain durable truth; browser memory holds only interaction/buffer state.
- No CDN, live model dependency, streaks or debt counters; use existing local assets and tokens.
- No hardcoded color literals outside tokens.css; keyboard/focus and reduced motion remain accessible.
- Tests use synthetic data or scratch repositories, never the real learner DB or workspace.
- Coordinator alone edits App.tsx, shared CSS/imports, package/lock config, Makefile and docs.

## Review Focus

- Rating a changed card must preserve the answer and recover from 409 without duplicate writes.
- Empty history/content, completed days and low energy must show honest actionable states.
- Closing or rating must update visible server-derived state without inventing solved mastery.
- Coverage and mastery are different dimensions; absent content is never shown as ready to study.
- Failed loads/writes and keyboard interactions must not lose notes or trigger unintended mutations.

### Task 1: Today and Review (I-16)

**Files:** create `web/src/api/learning.ts`, `web/src/pages/TodayPage.tsx`,
`web/src/pages/ReviewPage.tsx`, focused hooks/tests and optionally `styles/learning.css`.
**Interfaces:** typed API functions for GET today, POST session close, GET due reviews,
POST rating. Export `TodayPage` and `ReviewPage`; coordinator binds routes.

- [ ] Add failing behavior tests for last-note-first, exact block/time rendering,
  one first action, normal/low-energy requests, resume/day-complete/empty states,
  close note/answer save and failure preservation, due queue/reveal/rating/stale recovery.
- [ ] Implement Today with one clear start action, session blocks, transition note,
  low-energy control and close form. Link problem/review actions into the browser.
- [ ] Implement Review with bounded queue, Markdown QA/cloze text, explicit reveal,
  Again/Hard/Good/Easy, keys 1–4 only after reveal, disabled duplicate submits, and
  exact displayed due+content-hash preconditions. Recover cleanly from stale cards.
- [ ] Run focused tests, pnpm check/build; report RED/GREEN and limitations.

### Task 2: Map and Progress (I-17)

**Files:** create `web/src/api/overview.ts`, `web/src/pages/MapPage.tsx`,
`web/src/pages/ProgressPage.tsx`, focused hooks/tests and optionally `styles/overview.css`.
**Interfaces:** typed GET coverage and progress functions; export `MapPage`/`ProgressPage`.

- [ ] Add failing tests for tier/track filtering, content-vs-mastery labels, missing
  content, progress counts/week data, empty/error states and navigation to real problems.
- [ ] Implement a calm grouped topic map with accessible text labels plus mastery
  color cues. Use actual API coverage, never fabricate authored/ready state.
- [ ] Implement progress from server counts and mastery data; no invented retention
  estimates, streaks, comparisons or readiness beyond recorded states.
- [ ] Run focused tests, pnpm check/build; report evidence and remaining issues.

### Task 3: Integration, real-browser smoke and commits

**Files:** coordinator updates `web/src/App.tsx`, shared style imports,
`web/package.json`, lockfile, browser-test config/tests, Makefile and docs.

- [ ] Bind completed page exports and make Today the default first action.
- [ ] Add small Playwright smoke flows against `dogma serve` using a scratch repo:
  Today → Problem editor → save/run/submit → close → Review → Map/Progress.
  Verify page reload and a failed-save recovery case if practical.
- [ ] Verify generated API check, frontend checks/build, backend tests, e2e/slow
  suites without live model calls, privacy and the full exact-commit gate.
- [ ] Independent final review, fix material findings, update roadmap/handoff and
  commit I-16/I-17 in coherent pieces. Preserve unrelated content work throughout.
