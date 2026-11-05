import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { ReadinessOut } from "../../api/types";
import { ReadinessPanel } from "./ReadinessPanel";

const problemId = "cml/vector-operations/p01-vector-projection";
const readiness: ReadinessOut = {
  interview_ready: { ready: 0, total: 0, weeks: [] },
  next_due: [
    {
      problem_id: problemId,
      stage: 1,
      due_at: "2026-11-16T12:00:00Z",
      due_now: false,
      reason: "Second timed re-solve: 21 days after your last timed pass.",
    },
    {
      problem_id: "cml/vector-operations/p02-other",
      stage: 0,
      due_at: "2026-10-26T06:00:00Z",
      due_now: true,
      reason: "First timed re-solve: 7 days after your first pass.",
    },
  ],
  recent_results: [
    {
      problem_id: problemId,
      status: "passed",
      used_s: 872,
      budget_s: 1200,
      closed_at: "2026-10-26T09:14:00Z",
    },
  ],
};

it("lists the next due timed re-solves and the recent timed results", () => {
  render(<ReadinessPanel readiness={readiness} />);
  expect(screen.getByRole("heading", { name: "Readiness" })).toBeVisible();
  const due = screen.getByRole("list", { name: "Next timed re-solves" });
  expect(due).toHaveTextContent(`${problemId} · stage 1 of 3`);
  expect(due).toHaveTextContent("due now");
  const results = screen.getByRole("list", { name: "Last timed results" });
  expect(results).toHaveTextContent(`${problemId} · passed · 14 min 32 s of 20 min`);
});

it("says plainly when nothing is scheduled or timed yet", () => {
  render(
    <ReadinessPanel
      readiness={{
        next_due: [],
        recent_results: [],
        interview_ready: { ready: 0, total: 0, weeks: [] },
      }}
    />,
  );
  expect(
    screen.getByText(
      "No timed re-solve is scheduled yet. One is scheduled after you pass a core problem.",
    ),
  ).toBeVisible();
  expect(screen.getByText("No timed results yet.")).toBeVisible();
});

it("shows no streaks, debt or overdue counts", () => {
  const { container } = render(<ReadinessPanel readiness={readiness} />);
  expect(container.textContent ?? "").not.toMatch(/streak|overdue|debt|behind|missed/i);
});
