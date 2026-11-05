// Pure rules for the Mock page: calm elapsed time (never an alarm), and the score form's progress.
import type { Scores } from "./answers";

/** Items still without a whole score from 1 to 4. */
export function unscoredCount(criteria: readonly string[], scores: Scores): number {
  return criteria.filter((key) => {
    const value = scores[key];
    return value === undefined || !Number.isInteger(value) || value < 1 || value > 4;
  }).length;
}

/** How long the mock has run, in words. There is no limit to pass, so nothing turns red. */
export function elapsedLabel(startedAt: string, nowMs: number): string {
  const minutes = Math.floor((nowMs - Date.parse(startedAt)) / 60_000);
  if (!Number.isFinite(minutes) || minutes < 1) return "Just started";
  return `${minutes} min so far`;
}

/** A percentage with no decimals, for a part or the whole report. */
export function percentText(value: number): string {
  return `${Math.round(value)}%`;
}
