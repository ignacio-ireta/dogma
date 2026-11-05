import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import type { ResolveWindowOut, TimedResolveOut } from "../../api/types";
import { jsonResponse, queryWrapper } from "../../test/helpers";
import { TimedResolveAction } from "./TimedResolveAction";

const problemId = "cml/vector-operations/p01-vector-projection";
const timed: TimedResolveOut = {
  problem_id: problemId,
  title: "Vector projection",
  minutes: 20,
  open: false,
};
const window: ResolveWindowOut = {
  id: "scratch-window",
  problem_id: problemId,
  status: "open",
  started_at: "2026-10-26T09:00:00Z",
  budget_s: 1200,
  deadline: "2026-10-26T09:20:00Z",
  hints_used: 0,
  closed_at: null,
};

function failure(status: number, detail: string) {
  return jsonResponse(
    {
      type: "about:blank",
      title: "Conflict",
      status,
      detail,
      code: "RESOLVE_ALREADY_OPEN",
      instance: "",
      errors: null,
    },
    status,
  );
}

function renderAction(action: TimedResolveOut = timed) {
  return render(
    <MemoryRouter initialEntries={["/today"]}>
      <Routes>
        <Route path="/today" element={<TimedResolveAction timed={action} disabled={false} />} />
        <Route path="/problem/:track/:topic/:problem" element={<p>Problem page opened</p>} />
      </Routes>
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
}

afterEach(() => vi.unstubAllGlobals());

it("starts the timed re-solve and opens its Problem page", async () => {
  const bodies: unknown[] = [];
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    if (url.endsWith("/resolves/current")) return jsonResponse(null);
    bodies.push(JSON.parse(String(init?.body)));
    return jsonResponse({ window, previous_file_kept: false }, 201);
  });
  renderAction();
  const button = await screen.findByRole("button", { name: "Start timed re-solve" });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(await screen.findByText("Problem page opened")).toBeVisible();
  expect(bodies).toEqual([{ problem_id: problemId }]);
});

it("shows the API's refusal when another timed re-solve is open", async () => {
  vi.stubGlobal("fetch", async (url: string) =>
    url.endsWith("/resolves/current")
      ? jsonResponse(null)
      : failure(409, "A timed re-solve of cml/other/p01 is still open."),
  );
  renderAction();
  const button = await screen.findByRole("button", { name: "Start timed re-solve" });
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(button);
  expect(await screen.findByText("A timed re-solve of cml/other/p01 is still open.")).toBeVisible();
  expect(screen.queryByText("Problem page opened")).not.toBeInTheDocument();
});

it("continues the open window for this problem instead of starting another", async () => {
  vi.stubGlobal("fetch", async () => jsonResponse(window));
  renderAction();
  const link = await screen.findByRole("link", { name: "Open timed re-solve" });
  expect(link).toHaveAttribute("href", `/problem/${problemId}`);
  expect(screen.queryByRole("button", { name: "Start timed re-solve" })).not.toBeInTheDocument();
});

it("keeps the start button disabled while the session is busy", async () => {
  vi.stubGlobal("fetch", async () => jsonResponse(null));
  render(
    <MemoryRouter>
      <TimedResolveAction timed={timed} disabled />
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Start timed re-solve" })).toBeDisabled(),
  );
});
