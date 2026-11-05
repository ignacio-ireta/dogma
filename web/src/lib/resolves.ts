// Timed re-solve wording and arithmetic (ARCHITECTURE §4.4). The sentences match `dogma submit`
// and `dogma progress`, so the CLI and the browser describe a result the same way.
import type { ReSolveDueOut, TimedResultOut, TimedVerdictOut } from "../api/types";

/** Seconds as `14 min 32 s`, the form every timed result uses. */
export function clockText(seconds: number): string {
  const whole = Math.round(seconds);
  return `${Math.floor(whole / 60)} min ${String(whole % 60).padStart(2, "0")} s`;
}

/** Whole seconds from `nowMs` to the deadline, never negative. */
export function remainingSeconds(deadline: string, nowMs: number): number {
  return Math.max(0, Math.round((Date.parse(deadline) - nowMs) / 1000));
}

function budgetMinutes(budgetS: number): number {
  return Math.floor(budgetS / 60);
}

export function verdictLine(verdict: TimedVerdictOut): string {
  if (verdict.status === "passed") {
    return `Timed re-solve passed: ${clockText(verdict.used_s)} of ${budgetMinutes(verdict.budget_s)} min, no hints.`;
  }
  if (verdict.status === "failed") {
    return "Timed re-solve failed: not every test passed.";
  }
  const reasons: string[] = [];
  if (verdict.used_s > verdict.budget_s) {
    reasons.push(
      `over budget (${clockText(verdict.used_s)} of ${budgetMinutes(verdict.budget_s)} min)`,
    );
  }
  if (verdict.hints_used > 0) {
    reasons.push(`hints used (${verdict.hints_used})`);
  }
  const tail = verdict.hints_used > 0 ? ", so it counts as practice." : ".";
  return `Timed re-solve partial: ${reasons.join(" and ")}${tail}`;
}

/** The next due item in words. Due now is said as that, never as overdue. */
export function nextDueLine(due: ReSolveDueOut): string {
  const when = due.due_now ? "due now" : `due ${dateText(due.due_at)}`;
  return `Next timed re-solve: ${due.problem_id} ${when}, stage ${due.stage} of 3.`;
}

export function dueLine(due: ReSolveDueOut): string {
  const when = due.due_now ? "due now" : `due ${dateText(due.due_at)}`;
  return `${due.problem_id} · stage ${due.stage} of 3 · ${when}`;
}

export function timedResultLine(result: TimedResultOut): string {
  return `${result.status} · ${clockText(result.used_s)} of ${budgetMinutes(result.budget_s)} min`;
}

function dateText(iso: string): string {
  return new Date(iso).toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
