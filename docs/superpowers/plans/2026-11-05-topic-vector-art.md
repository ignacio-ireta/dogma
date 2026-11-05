# Topic vector artwork implementation

**Goal:** Replace the production generated image with hand-authored Memphis figures and topic art.

**Architecture:** Reusable React SVG figure and motif components render a large study scene or
compact lesson-header scene. A pure explicit topic mapping chooses the motif from an existing ID.

**Stack:** Existing React/TypeScript/CSS, Vitest and Playwright; no dependencies.

**Spec:** [Topic vector artwork](../specs/2026-11-05-topic-vector-art.md).

**Execution:** Root draws and integrates the vectors; Luna handles taxonomy coverage and review.
The user's existing implementation authorization applies to this requested visual revision.

## Constraints and review focus

- Preserve the uncommitted prior typography/Markdown/theme changes.
- Unknown/null topics must not pretend to show a known subject.
- All authored topic IDs must have a deliberate assignment; new content gets a failing coverage
  test until its illustration is assigned.
- Artwork must stay decorative and must not introduce focus stops, SVG ID collisions or images.
- Long titles and 320px layouts must fit without squeezing text or widening the page.
- Today must update with its actual session; workspace tabs, math and execution must still work.

## Tasks

- [x] Add `web/src/lib/topicIllustration.ts` and its tests; check coverage against local packages.
  Interface: `topicScene(topicId?: string | null): TopicScene`.
- [x] Draw SVG figures and subject motifs in `web/src/components/illustrations/`; style using
  tokens in `web/src/styles/illustrations.css`. Interface:
  `StudyIllustration({ topicId?, compact? })`.
- [x] Replace the bitmap in PageHeading, wire Today/current topic and ProblemPage/topic ID;
  remove the obsolete production bitmap and document where the editable drawings live.
- [x] Inspect rendered desktop/mobile scenes and representative subjects, run mapping tests,
  frontend checks/build and relevant Playwright flows; fix any visual or behavior regressions.

## Verification outcome

- All 175 authored packages explicitly mapped (168 with exercise pages); null/unknown IDs use
  the general book scene. TypeScript checks that all 21 scene names have an SVG motif.
- Biome, TypeScript, 187 unit tests, API-type freshness, and production build passed. The API
  freshness command was run separately with cache access after the sandbox blocked uv's cache.
- 22 browser tests passed: two new illustration tests and all 20 existing browser regressions.
  The disposable server now includes real SQL and tokenizer packages for navigation coverage.
- Eight main routes at 1440/390/320px: no horizontal page overflow or browser/runtime errors.
  Reading font, math, scrollable tables, white workspace panels and contrast checks passed.
- Visually reviewed all 21 scenes together and the compact 320px lesson header; no clipping or
  unintended overlaps. Read-only Luna review found no material issues.
- [Offline vector gallery](../../design/2026-11-05-topic-vectors/index.html) captures the final
  drawings for review. The React SVG components are the editable source of truth.
- No generated raster artwork remains in production source, public assets or built assets.
  The existing Vite large-chunk warning remains unchanged in kind.
