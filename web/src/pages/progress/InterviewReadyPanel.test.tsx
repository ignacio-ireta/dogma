import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { InterviewReadyOut } from "../../api/types";
import { InterviewReadyPanel } from "./InterviewReadyPanel";

const ready: InterviewReadyOut = {
  ready: 1,
  total: 25,
  weeks: [
    { week: 1, ready: 1, total: 12 },
    { week: 2, ready: 0, total: 13 },
  ],
};

it("counts interview-ready topics in total and per study week", () => {
  render(<InterviewReadyPanel ready={ready} />);
  expect(screen.getByRole("heading", { name: "Interview-ready" })).toBeVisible();
  expect(screen.getByText("1 of 25 core topics are interview-ready.")).toBeVisible();
  const list = screen.getByRole("list", { name: "Interview-ready topics per week" });
  expect(list).toHaveTextContent("Week 1 · interview-ready 1 of 12 topics");
  expect(list).toHaveTextContent("Week 2 · interview-ready 0 of 13 topics");
});

it("shows no streaks, trends or backlog", () => {
  const { container } = render(<InterviewReadyPanel ready={ready} />);
  expect(container.textContent ?? "").not.toMatch(/streak|trend|overdue|debt|behind|missed/i);
});
