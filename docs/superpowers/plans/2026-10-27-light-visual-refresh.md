# Light Visual Refresh Implementation Plan

**Goal:** Make the browser app inviting and calm, with a light theme and OpenDyslexic as the
default interface and reading font, as requested by the learner.

**Design direction:** Warm paper, white cards, slate ink and muted blue accents. Give the
navigation a clear identity, establish consistent page introductions, and make the daily plan
feel like a sequence of manageable steps. Keep semantic status colors and visible focus states.

**Approach:** Update the shared design tokens, shell and existing components in place. This
keeps the same React routes, API and study interactions. A new framework or separate visual
prototype would duplicate the existing app and delay an immediately usable result. The
checkout includes in-progress learning pages; preserve and style that work directly.

**Tech stack:** Existing React, TypeScript, Vite, CSS, CodeMirror and Playwright. Vendor the
upstream OpenDyslexic WOFF2 assets and license; no new application dependencies or runtime CDN.

## Tasks

- [x] Bundle regular, bold and italic OpenDyslexic fonts with pinned source attribution.
- [x] Replace automatic dark colors with a deliberate warm light palette. Improve spacing,
  reading measure, hierarchy, borders, focus rings, controls and code presentation.
- [x] Refresh the shared shell and navigation with lightweight SVG icons, clear active
  states and a responsive layout. Add consistent page headings and restrained decoration.
- [x] Style Today, Review, Problems, Map and Progress consistently. Emphasize the first action
  and retain honest data, existing accessible names and all study behaviors.
- [x] Reproduce and resolve the existing Today-page instruction regression while touching
  that screen; keep its existing failing assertion and fix the punctuation boundary.
- [x] Run frontend checks and the production build. Use the existing disposable-data browser
  fixture to verify fonts, dark-system behavior, mobile layouts and the learning workflow.
- [x] Inspect real desktop and mobile screenshots, adjust visual issues, then record the
  changes and verification in the roadmap and architecture documentation.

## Review focus

- OpenDyslexic must actually load, including bold and italic faces, without outside requests.
- A dark operating-system preference must still produce the requested light appearance.
- Long titles and filenames, larger text and narrow viewports must not hide controls or
  cause page-level horizontal scrolling. Code and wide tables may scroll within their panels.
- Keyboard navigation, skip link, focus indicators, review ratings, editor shortcuts and
  reduced-motion support must survive the styling changes.
- Verify using temporary learner data; do not create or modify real progress for screenshots.

## Verification

`make check` passed (993 Python tests, 89 topics, four import contracts, privacy, 94 frontend
tests, API consistency and production build). Both Playwright learning-workflow tests passed.
A real-browser visual check loaded all four OpenDyslexic faces, kept light colors under a dark
system preference, and visited six screens at 320, 390 and 768 pixels without page overflow,
outside requests or browser errors. Independent review verified 200% text at 320 pixels,
long titles, shortcut popovers, keyboard focus and retained table semantics. The final frontend
gate, production build and desktop/mobile captures passed after the last focus adjustment.

The Today regression had contradictory expectations: an early assertion required the raw
lesson path, while the detailed browser-instruction test required "the Lesson tab" for the
same first action. The implementation now handles sentence-ending punctuation, and the stale
assertion checks the explicit browser phrase. Assertions and detailed goal coverage remain.
