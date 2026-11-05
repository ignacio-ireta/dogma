import { describe, expect, it } from "vitest";
import type { ReSolveDueOut, TimedVerdictOut } from "../api/types";
import { clockText, nextDueLine, remainingSeconds, timedResultLine, verdictLine } from "./resolves";

const problemId = "cml/vector-operations/p01-vector-projection";

function verdict(overrides: Partial<TimedVerdictOut>): TimedVerdictOut {
  return {
    problem_id: problemId,
    status: "passed",
    used_s: 872,
    budget_s: 1200,
    hints_used: 0,
    next_due: null,
    ...overrides,
  };
}

describe("clockText", () => {
  it("reads seconds the way the CLI does: minutes, then two-digit seconds", () => {
    expect(clockText(872)).toBe("14 min 32 s");
    expect(clockText(0)).toBe("0 min 00 s");
    expect(clockText(59.6)).toBe("1 min 00 s");
  });
});

describe("remainingSeconds", () => {
  it("counts down to the deadline and never goes below zero", () => {
    const deadline = "2026-10-26T09:20:00Z";
    expect(remainingSeconds(deadline, Date.parse("2026-10-26T09:05:00Z"))).toBe(900);
    expect(remainingSeconds(deadline, Date.parse("2026-10-26T09:25:00Z"))).toBe(0);
  });
});

describe("verdictLine", () => {
  it.each([
    [verdict({}), "Timed re-solve passed: 14 min 32 s of 20 min, no hints."],
    [
      verdict({ status: "partial", used_s: 1500 }),
      "Timed re-solve partial: over budget (25 min 00 s of 20 min).",
    ],
    [
      verdict({ status: "partial", hints_used: 2 }),
      "Timed re-solve partial: hints used (2), so it counts as practice.",
    ],
    [
      verdict({ status: "partial", used_s: 1500, hints_used: 1 }),
      "Timed re-solve partial: over budget (25 min 00 s of 20 min) and hints used (1), so it counts as practice.",
    ],
    [verdict({ status: "failed" }), "Timed re-solve failed: not every test passed."],
  ])("says the verdict in words for %#", (value, text) => {
    expect(verdictLine(value)).toBe(text);
  });
});

describe("nextDueLine", () => {
  const due: ReSolveDueOut = {
    problem_id: problemId,
    stage: 1,
    due_at: "2026-11-16T12:00:00Z",
    due_now: false,
    reason: "Second timed re-solve: 21 days after your last timed pass.",
  };

  it("names the problem, the date and the stage", () => {
    expect(nextDueLine(due)).toMatch(
      new RegExp(`^Next timed re-solve: ${problemId} due .*2026.*, stage 1 of 3\\.$`),
    );
  });

  it("says a re-solve that is due is due now, not overdue", () => {
    const text = nextDueLine({ ...due, due_now: true });
    expect(text).toBe(`Next timed re-solve: ${problemId} due now, stage 1 of 3.`);
    expect(text).not.toMatch(/overdue|late|behind/i);
  });
});

describe("timedResultLine", () => {
  it("shows the verdict with the time used against the budget", () => {
    expect(
      timedResultLine({
        problem_id: problemId,
        status: "partial",
        used_s: 1500,
        budget_s: 1200,
        closed_at: "2026-10-26T09:25:00Z",
      }),
    ).toBe("partial · 25 min 00 s of 20 min");
  });
});
