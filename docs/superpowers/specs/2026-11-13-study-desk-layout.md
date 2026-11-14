# Study desk: layout refinement of the Memphis theme

Requested on 13 November 2026 as a refinement, not a rebrand. The palette, the illustrations, the
fonts and every study flow of `2026-11-05-memphis-theme-design.md` stay. What changes is how much
of the page is put in a box.

## The rule

The page is a study desk. Things that are objects on a desk keep a frame; writing does not.

| Framed (objects) | Open on the page (writing) |
| --- | --- |
| Code editor, the reading sheet beside it, the workbench | Introductions and explanations |
| Forms: the close note, answers, mock scoring | Lists of topics, problems, places and terms |
| The last transition note (a note left on the desk) | Headline figures and readiness lists |
| The review card, a timed re-solve window, dialogs | The session plan and its table of blocks |
| The first-action callout, notices, buttons, selects | Filters (the selects are framed, their row is not) |
| Illustrations and the session bar | Page headings |

Open content is organised with an ink rule (`--rule-ink`) between sections, hairlines
(`--rule`) between rows, and, where the width allows, a margin column holding the section number
and name (`--margin-col`). Prose keeps a measure of `--measure`.

## Where this narrows the Memphis specification

The Memphis specification asked for "bold outlines, offset shadows, and varied card surfaces" on
Today and Start here. Those surfaces remain on both pages, on objects only: the transition note,
the close form, the active navigation item, the primary button and the session bar. The tinted
cards that wrapped paragraphs, the five session blocks, the six places and the session table rows
are now ruled lists. The three tint colours moved to the session bar's segments.

## The session bar

`MinutesBar` draws a session's blocks to scale: each segment is as wide as its share of the
minutes, and segment *n* is row *n* of the list or table below it. It uses the real blocks (the
fixed Learn shape on Start here, the planned session from the API on Today), so it changes when
the plan does. The caption states the total and that the bar is to scale. It is hidden from
assistive technology because the rows carry the same facts as text.

## Typography

Unchanged, by the standing rule in ENGINEERING_STANDARDS §7: OpenDyslexic Bold for the interface,
Inter for rendered Markdown, KaTeX for mathematics, JetBrains Mono for code. JetBrains Mono now
also sets instrument output: minutes, counts, dates, topic ids and test labels. Labels stay in
sentence case; no small capitals or tracked uppercase labels were introduced, because they are
harder to read and the interface font was chosen for readability.

## Motion

Illustrations stay static. Two functional motions exist: the session bar is drawn once, left to
right, in 700 ms; the "Running your file" line breathes slowly while the runner is working. Both
stop under `prefers-reduced-motion` through the global override in `tokens.css`.

## Not changed

Routes, API contracts, copy meaning, grading, keyboard shortcuts, the workbench layout and the
lesson typography. The Problem page keeps its two white panels with quiet borders.
