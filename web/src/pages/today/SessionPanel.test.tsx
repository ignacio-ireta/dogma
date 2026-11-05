import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import type { SessionOut, TodayOut } from "../../api/learning";
import { jsonResponse, queryWrapper } from "../../test/helpers";
import { SessionPanel } from "./SessionPanel";

const problemId = "cml/vector-operations/p01-vector-projection";
const session: SessionOut = {
  session_id: "scratch-session",
  plan_date: "2026-10-26",
  slot: 2,
  sessions_per_day: 3,
  kind: "consolidate",
  energy: "normal",
  topic_id: null,
  must_do: null,
  first_action: `Start the timed re-solve: dogma resolve ${problemId} (budget 20 min, no hints).`,
  minutes: 45,
  notes: [],
  blocks: [
    {
      name: "Re-solve",
      minutes: 25,
      detail: `Timed re-solves, no hints: Vector projection (${problemId}), budget 20 min. Start with dogma resolve ${problemId}, then dogma submit ${problemId} when you have a solution.`,
    },
  ],
  timed_resolve: { problem_id: problemId, title: "Vector projection", minutes: 20, open: false },
};
const view = {
  local_date: "2026-10-26",
  sessions_per_day: 3,
  closed_today: 1,
  last_note: null,
  current: session,
  resumed: false,
  replanned: false,
  stale_open: [],
} as unknown as TodayOut;

afterEach(() => vi.unstubAllGlobals());

it("makes a planned timed re-solve the session's start action, not a plain link", async () => {
  vi.stubGlobal("fetch", async () => jsonResponse(null));
  render(
    <MemoryRouter>
      <SessionPanel view={view} session={session} disabled={false} />
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Start timed re-solve" })).toBeEnabled(),
  );
  expect(screen.queryByRole("link", { name: /Start session/ })).not.toBeInTheDocument();
  expect(screen.getByRole("table", { name: "Your session blocks" })).toHaveTextContent(
    "Start on the Problem page, then Submit when you have a solution.",
  );
});

it("keeps the ordinary start link for a session without a timed re-solve", () => {
  const plain = { ...session, timed_resolve: null, first_action: "Run dogma review --limit 5." };
  render(
    <MemoryRouter>
      <SessionPanel view={view} session={plain} disabled={false} />
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
  expect(screen.queryByRole("button", { name: "Start timed re-solve" })).not.toBeInTheDocument();
});
