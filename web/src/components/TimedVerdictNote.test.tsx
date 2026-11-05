import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { TimedVerdictOut } from "../api/types";
import { TimedVerdictNote } from "./TimedVerdictNote";

const verdict: TimedVerdictOut = {
  problem_id: "cml/vector-operations/p01-vector-projection",
  status: "passed",
  used_s: 872,
  budget_s: 1200,
  hints_used: 0,
  next_due: {
    problem_id: "cml/vector-operations/p01-vector-projection",
    stage: 1,
    due_at: "2026-11-16T12:00:00Z",
    due_now: false,
    reason: "Second timed re-solve: 21 days after your last timed pass.",
  },
};

it("says the verdict in words and when the next timed re-solve is due", () => {
  render(<TimedVerdictNote verdict={verdict} />);
  expect(screen.getByText("Timed re-solve passed: 14 min 32 s of 20 min, no hints.")).toBeVisible();
  expect(screen.getByText(/Next timed re-solve: .* due .*2026.*stage 1 of 3\./)).toBeVisible();
});

it("shows the partial reason when the learner hinted", () => {
  render(
    <TimedVerdictNote verdict={{ ...verdict, status: "partial", hints_used: 1, next_due: null }} />,
  );
  expect(
    screen.getByText("Timed re-solve partial: hints used (1), so it counts as practice."),
  ).toBeVisible();
  expect(screen.queryByText(/Next timed re-solve/)).not.toBeInTheDocument();
});
