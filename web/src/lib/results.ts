// Plain wording for a graded attempt. Pure: the results panel renders these strings, so the
// wording and the counts are tested without a DOM.
import type { AttemptCreated, LearnerReport, LearnerTest } from "../api/types";

export type RunStatus = LearnerReport["status"];

const VERDICTS: Record<RunStatus, string> = {
  passed: "All tests passed",
  failed: "Not passed yet",
  timeout: "Stopped at the time limit",
  oom: "Stopped at the memory limit",
  error: "The tests could not run",
  rejected: "Not run: fix these first",
};

export function verdictLine(status: RunStatus): string {
  return VERDICTS[status];
}

export interface Counts {
  readonly passed: number;
  readonly total: number;
}

export function countsFor(tests: readonly LearnerTest[], file: LearnerTest["file"]): Counts {
  const mine = tests.filter((test) => test.file === file);
  return {
    passed: mine.filter((test) => test.outcome === "pass").length,
    total: mine.length,
  };
}

export interface Mark {
  readonly symbol: string;
  readonly label: string;
  readonly tone: "pass" | "fail" | "skip";
}

export function markFor(outcome: LearnerTest["outcome"]): Mark {
  switch (outcome) {
    case "pass":
      return { symbol: "✓", label: "passed", tone: "pass" };
    case "skip":
      return { symbol: "–", label: "skipped", tone: "skip" };
    case "fail":
      return { symbol: "✗", label: "failed", tone: "fail" };
    case "error":
      return { symbol: "✗", label: "errored", tone: "fail" };
  }
}

/** Whether the attempt reached the record, in one calm sentence. */
export function recordedLine(result: AttemptCreated): string {
  if (result.recorded) {
    return "Recorded in your history.";
  }
  return result.record_warning
    ? `Not recorded: ${result.record_warning}`
    : "Not recorded in your history.";
}
