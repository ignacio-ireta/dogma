# Memphis Theme Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement these tasks. User authorized Luna agents for bounded legwork and review.

**Goal:** Apply the selected Memphis Study Club design with stronger color contrast, a less yellow canvas, and quiet lesson workspaces.

**Architecture:** Extend the current semantic CSS tokens and existing shared components. Add an optional illustrated PageHeading variant for Today/Welcome; retain every route and action.

**Tech Stack:** React, TypeScript, Vite, CSS, existing Vitest/Playwright suites.

**Spec:** `docs/superpowers/specs/2026-11-05-memphis-theme-design.md`

## Constraints and review focus

- Preserve earlier uncommitted typography/Markdown work on the existing feature branch.
- No new dependencies, navigation restructuring, API changes, or learner-data mutations.
- Preserve semantic status hues and bold interface/Inter reading font split.
- Check 320px, 390px, and desktop widths; long headings and tables must stay usable.
- Check active/focus controls, quiet lesson/code surfaces, and actual rendered contrast.

## Task 1: Palette and shared visual system

- [x] Extend `web/src/styles/tokens.css` with neutral surfaces and separate coral/violet/navy decoration tokens.
- [x] Update `web/src/styles/app.css` shell, navigation, heading, buttons, cards and quiet problem workspace. Keep existing responsive behavior.
- [x] Apply selected card accents in `learning.css`, `welcome.css`, and `overview.css` without changing component behavior.
- [x] Calculate and check text/control contrast using rendered colors.

## Task 2: Original illustration and page integration

- [x] Edit the existing original illustration with the built-in image tool: stronger coral/violet/cobalt and a neutral background, preserving its figures/composition. Encode losslessly as WebP and save it with provenance under `web/public/illustrations/`.
- [x] Add optional illustrated heading treatment to `PageHeading.tsx`, enabled only in `TodayPage.tsx` and `WelcomePage.tsx`.
- [x] Use curiosity-focused shell/header copy while preserving content, actions and page titles.
- [x] Inspect Today, Welcome, and the vector-projection lesson visually on desktop and phone.

## Task 3: Verification and documentation

- [x] Run `pnpm --dir web check` and `pnpm --dir web build`; expect lint/types/unit tests/API generation check and build success.
- [x] Run existing welcome, learning and Markdown browser tests against disposable learner data.
- [x] Audit every main route at desktop, 390px and 320px for overflow, runtime/asset errors, fonts and contrast. Do not interact with real learner data.
- [x] Review the final diff and rendered screenshots for regressions. Luna completed the initial component audit and integration; the parent completed final review after the reviewer hit its usage limit.
- [x] Update design/architecture documentation and give the user a running preview or saved screenshots.

This is a visual change: use existing behavior tests and browser/rendered-contrast checks instead of adding tests that merely assert CSS implementation details.

## Verification outcome

- Lint and TypeScript passed; 184 unit tests passed. The API type check passed after rerunning that command with access to the existing uv cache.
- Production build passed. Its pre-existing large-JavaScript-chunk warning remains; this theme introduces no dependency.
- 13 existing browser tests passed: full learner session/save/run/submit/review flow, welcome/navigation/keyboard behavior, and Markdown/math/table rendering at desktop and phone widths.
- The cross-page browser audit found Vite embedding two small fonts as data URLs, which the API's existing self-only font CSP blocks. A new browser regression test failed for the Cyrillic code-font subset, then passed after `assetsInlineLimit: 0` made font assets local files. The security policy is unchanged.
- Final audit: eight routes at 1440/390/320px, loaded fonts and images, quiet white lesson panels, no page overflow, no runtime/console/asset errors. Sampled actual text contrast: 6.12:1 to 13.98:1. Thirty-five palette pairs also passed their 4.5:1 text/3:1 control thresholds.
- The initial transparent artwork edit introduced unwanted haze. The selected original-based edit uses a solid near-neutral background, then lossless WebP encoding (736,636 bytes); no rejected candidate is shipped.
- No API, navigation, planning, execution, or persistence behavior changed. The earlier authorized typography and Markdown work remains intact.

Screenshots captured during review: `/private/tmp/dogma-memphis-today-desktop.png`, `/private/tmp/dogma-memphis-today-phone.png`, `/private/tmp/dogma-memphis-welcome.png`, `/private/tmp/dogma-memphis-lesson.png`.
