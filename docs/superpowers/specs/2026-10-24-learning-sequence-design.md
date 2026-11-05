# Learning sequence design

The user approved the continuation sequence on 2026-10-24. The existing
ARCHITECTURE, CONTENT_SPEC, ENGINEERING_STANDARDS and LEARNER_PROFILE are the
binding product specification. This document records implementation decisions
needed to execute that sequence without restarting product discovery.

## Intent

Complete the daily practice loop: remember studied material, choose today's
work, and leave a useful next step. Preserve the established CLI, content ids,
learner state and original reference material. Work remains local and offline
except explicit requests to the loopback model gateway.

## Approaches

The approved approach finishes FSRS and daily sessions first, maintains content
runway, then adds retrieval, the tutor, and the API/browser milestones. Advancing
the browser first would leave the learning behavior incomplete. Producing more
weeks before the learning engine would lengthen an already adequate buffer
without making practice easier to start. Continue week-4 review and week-5 drafts
alongside platform work because their owned paths are independent.

## Cards and reviews (I-09)

- A small FSRS adapter uses target retention 0.90, explicit aware timestamps,
  and disabled fuzzing so behavior is reproducible.
- New `card` and `card_review` tables arrive through a forward-only migration.
  Stable content ids identify cards; content hashes detect edits. Edits retain
  scheduling/history. Missing or retired content becomes inactive, never deleted.
- Only verified/reviewed topics with learner state at least introduced enter
  scheduling. Existing solved topics are enrolled on synchronization. Authored
  but unseen topics never create a review backlog.
- `dogma review` synchronizes eligible cards, presents a bounded due queue,
  reveals answers, accepts Again/Hard/Good/Easy, and persists each rating. EOF or
  quit before rating does not record a review. QA and cloze cards are supported.
- Reviewing early, invalid ratings, invalid timestamps and stale/repeated rating
  submissions are rejected clearly. A review event and its schedule update are
  atomic. Reviews cannot promote a topic directly to retained/interview-ready.
- A good review at least three days after solving can earn consolidating. A
  lapse from a later state returns to consolidating without erasing milestones.

## Sessions (I-10)

- A deterministic planner consumes ready content, prerequisites, topic state,
  due reviews, path order, local date and energy. Time is passed explicitly in
  tests. Ordinary days have Learn + Learn + Consolidate sessions of 45 minutes.
- The Learn structure is Retrieve 5 / Predict 3 / Learn 12 / Practice 20 / Close
  5 minutes. Consolidate is Review 15 / re-solve 25 / Close 5. Low-energy sessions
  are Review 10 / familiar re-solve 10, introduce no new topics, and have a clear
  empty-state fallback. Planned prerequisite completion is never treated as an
  actual solve.
- `dogma today` resumes an existing open session, displays the last transition
  note and one first action, and stores plans in SQLite. Repeated invocation must
  not duplicate sessions or introduce new material implicitly.
- `dogma close` stores where the learner stopped, the next step, an optional
  blocker, and a self-written answer. Closing is idempotent and never invents a
  successful solve. It enrolls introduced topic cards through I-09.
- Timed retention/readiness remains I-13; sessions may suggest familiar practice
  now, without claiming timed mastery that has not been implemented.

## Content and subsequent increments

Review every week-4 topic using ENGINEERING_STANDARDS §5.6, correcting defects
without weakening tests and bumping problem versions when contracts/tests change.
Finish all thirteen week-5 drafts, verify their solutions and starters, and get
independent review. Preserve all existing authored work. Keep future content at
least one week ahead; week-6 authoring follows this wave.

I-11 builds heading/page-aware chunks and a local FTS5 index using the vendored
docmd tool. I-12 adds the local/fake gateway, bounded context, versioned prompts,
cited hint/debug responses and graceful unavailability. I-14 through I-17 expose
the same services through a versioned FastAPI API and an offline React UI, with
no separate browser truth for learner state. No hosting or publishing is needed.

I-20 aligns the complete test gate and adds hooks. I-21 hardens execution and
retention without deleting learner attempts. I-19 adds the optional torch runtime
before week-8 authoring (deadline 30 November). Each receives a focused plan and
acceptance tests when picked up; this does not authorize unrelated redesigns.

## Verification and boundaries

All development databases and runs use scratch settings. Never read private
learner data or mutate real progress for testing. Preserve no-record semantics.
Use tests that exercise behavior and migration data preservation; run the full
gate before claiming completion. Content drafts initially prevent a green content
gate and must be finished, not excluded or weakened to hide them.
