# 0019. A personal course calendar (time-warp)

Date: 2026-11-03 · Status: accepted

## Context

The course calendar in `curriculum/paths/core.yaml` is fixed (26 Oct 2026 to 31 Jan 2027). Topic
order already follows progress, but the week number, the weekly mocks from week 3 and the
interview-only final week follow the date. After a break or a busy stretch the calendar moves on
without the learner, mocks arrive before their material, and the interview week can start with
topics still unlearned. Editing the YAML by hand would mix learner state into authored content.

## Decision

Keep the authored calendar. Store a personal offset in whole weeks in SQLite (`calendar_warp`, an
append-only log; the newest row is in force). Every date-dependent reader loads the shifted path.
A sync makes this week the week of the furthest core-path exercise the learner has run. The
learner can also pick a week, move ahead or back, or restore the authored dates. A separate course
reset backs up, clears progress and makes this week week 1. Card and re-solve due dates do not
move.

## Consequences

Breaks never leave the calendar ahead of the work, and the end date moves with it. The interview
week can land after the original end date. Review due dates still reflect real elapsed time, so a
long break brings due reviews, which the planner already caps and spreads. A reset deletes rows,
outside any migration, only after a successful backup and a typed confirmation.
