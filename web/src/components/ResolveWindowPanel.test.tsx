import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { ResolveWindowOut } from "../api/types";
import { jsonResponse, queryWrapper } from "../test/helpers";
import { ResolveWindowPanel } from "./ResolveWindowPanel";

const window: ResolveWindowOut = {
  id: "scratch-window",
  problem_id: "cml/vector-operations/p01-vector-projection",
  status: "open",
  started_at: "2026-10-26T09:00:00Z",
  budget_s: 1200,
  deadline: "2026-10-26T09:20:00Z",
  hints_used: 0,
  closed_at: null,
};

beforeEach(() => {
  vi.useFakeTimers({
    toFake: ["Date", "setInterval", "clearInterval", "setTimeout", "clearTimeout"],
  });
  vi.setSystemTime(Date.parse("2026-10-26T09:05:00Z"));
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function panel(value: ResolveWindowOut = window) {
  return render(<ResolveWindowPanel window={value} />, { wrapper: queryWrapper() });
}

it("shows the budget, the time left and the rules", () => {
  panel();
  expect(screen.getByRole("heading", { name: "Timed re-solve open" })).toBeVisible();
  expect(screen.getByText("Time budget 20 min")).toBeVisible();
  expect(screen.getByText("15 min 00 s left")).toBeVisible();
  expect(screen.getByText(/Run is unlimited/)).toBeVisible();
  expect(screen.getByText(/The first submit decides it/)).toBeVisible();
  expect(screen.getByText(/Hints count as practice/)).toBeVisible();
});

it("counts down calmly as time passes", () => {
  panel();
  act(() => {
    vi.advanceTimersByTime(60_000);
  });
  expect(screen.getByText("14 min 00 s left")).toBeVisible();
});

it("tells the learner the budget has passed without alarm styling or an alert role", () => {
  vi.setSystemTime(Date.parse("2026-10-26T09:25:00Z"));
  const { container } = panel();
  expect(screen.getByText(/The budget has passed/)).toBeVisible();
  expect(screen.getByText(/Submit when you are ready/)).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(container.querySelector(".resolve--over")).not.toBeNull();
  expect(container.querySelector(".notice--error")).toBeNull();
});

it("abandons only after the learner confirms", async () => {
  const requests: string[] = [];
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    requests.push(`${init?.method ?? "GET"} ${url}`);
    return jsonResponse({ ...window, status: "abandoned", closed_at: "2026-10-26T09:05:00Z" });
  });
  panel();
  fireEvent.click(screen.getByRole("button", { name: "Abandon re-solve" }));
  const dialog = screen.getByRole("alertdialog", { name: "Abandon this timed re-solve?" });
  expect(dialog).toHaveTextContent("No penalty");
  expect(requests).toEqual([]);
  fireEvent.click(screen.getByRole("button", { name: "Keep going" }));
  expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  expect(requests).toEqual([]);
  fireEvent.click(screen.getByRole("button", { name: "Abandon re-solve" }));
  fireEvent.click(screen.getByRole("button", { name: "Yes, abandon" }));
  await vi.waitFor(() => expect(requests).toEqual(["POST /api/v1/resolves/current/abandon"]));
});
