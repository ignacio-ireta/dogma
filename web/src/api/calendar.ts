// The personal course calendar (ADR-0019): read it, preview or move it, or start the course over.
// A move is previewed with dry_run, then confirmed with dry_run false. Nothing is written by a preview.
import { apiFetch } from "./client";
import type { components } from "./schema";

type Schemas = components["schemas"];
export type CalendarOut = Schemas["CalendarOut"];
export type CalendarWeekOut = Schemas["CalendarWeekOut"];
export type CalendarWarpRequest = Schemas["CalendarWarpRequest"];
export type CalendarWarpOut = Schemas["CalendarWarpOut"];
export type CalendarPointOut = Schemas["CalendarPointOut"];
export type ResetOut = Schemas["ResetOut"];
/** A move as the learner asks for it. The dry-run flag is set by previewWarp or applyWarp. */
export type CalendarMove = Omit<CalendarWarpRequest, "dry_run">;

export function getCalendar(): Promise<CalendarOut> {
  return apiFetch<CalendarOut>("/calendar");
}

/** What a move would do. Writes nothing. */
export function previewWarp(request: CalendarMove): Promise<CalendarWarpOut> {
  return warp({ ...request, dry_run: true });
}

/** Make the move. The API writes one row only when the offset changes. */
export function applyWarp(request: CalendarMove): Promise<CalendarWarpOut> {
  return warp({ ...request, dry_run: false });
}

function warp(body: CalendarWarpRequest): Promise<CalendarWarpOut> {
  return apiFetch<CalendarWarpOut>("/calendar/warp", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Back up, clear all progress and make this week week 1. Only the literal word `reset` is accepted. */
export function startOver(): Promise<ResetOut> {
  return apiFetch<ResetOut>("/course/reset", {
    method: "POST",
    body: JSON.stringify({ confirm: "reset" }),
  });
}
