import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CalendarOut, CalendarWarpOut } from "../../api/calendar";
import { jsonResponse, queryWrapper } from "../../test/helpers";
import { CalendarOffer } from "../today/CalendarOffer";
import { CalendarPanel } from "./CalendarPanel";
import { dayLabel } from "./calendarText";
import { StartOverPanel } from "./StartOverPanel";

interface Scratch {
  current: number;
  anchor: number;
  offset: number;
}

function calendarBody(state: Scratch): CalendarOut {
  const weeks = Array.from({ length: 14 }, (_, index) => {
    const starts = new Date(Date.UTC(2026, 9, 26 + index * 7));
    return {
      week: index + 1,
      starts: starts.toISOString().slice(0, 10),
      theme: `Theme ${index + 1}`,
    };
  });
  return {
    offset_weeks: state.offset,
    current_week: state.current,
    anchor_week: state.anchor,
    anchor_problem_id: state.anchor === 1 ? null : "cml/gradient-descent/p01-step",
    start: "2026-10-26",
    end: state.offset === 0 ? "2027-01-31" : "2027-02-07",
    weeks,
    last_warp: null,
  };
}

function warpBody(state: Scratch, target: number, applied: boolean): CalendarWarpOut {
  const point = (week: number, offset: number) => ({
    offset_weeks: offset,
    current_week: week,
    current_starts: "2026-10-26",
    mocks_from: "2026-11-09",
    interview_week: "2027-01-25",
    end: "2027-01-31",
  });
  return {
    applied,
    changed: target !== state.current,
    before: point(state.current, state.offset),
    after: point(target, state.offset),
    anchor_week: state.anchor,
    anchor_problem_id: null,
  };
}

/** A scratch API: the calendar, a warp that records each body, and the reset. */
function serve(state: Scratch = { current: 1, anchor: 1, offset: 0 }) {
  const posts: Record<string, unknown>[] = [];
  vi.stubGlobal("fetch", async (path: string, init?: RequestInit) => {
    const url = new URL(path, "http://localhost");
    if (url.pathname.endsWith("/calendar/warp")) {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      posts.push(body);
      const target = body.kind === "sync" ? state.anchor : Number(body.week);
      const applied = body.dry_run === false;
      if (applied) {
        state.current = target;
        state.offset = target - 1;
      }
      return jsonResponse(warpBody(state, target, applied));
    }
    if (url.pathname.endsWith("/course/reset")) {
      posts.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
      return jsonResponse({
        backup_path: "/scratch/var/backups/dogma-backup-20261103.json",
        snapshot_path: "/scratch/var/backups/dogma-snapshot-20261103.db",
        rows_cleared: { attempt: 1 },
        files_set_aside: 0,
        history_dir: null,
        workspace_error: null,
        week_one_starts: "2026-11-02",
        course_end: "2027-02-07",
      });
    }
    if (url.pathname.endsWith("/calendar")) {
      return jsonResponse(calendarBody(state));
    }
    return jsonResponse({ detail: "not served" }, 404);
  });
  return posts;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CalendarPanel", () => {
  it("previews a move with dry_run, then the confirmation makes it with dry_run false", async () => {
    const posts = serve();
    const user = userEvent.setup();
    render(<CalendarPanel />, { wrapper: queryWrapper() });

    await screen.findByText(/This week is week 1 of 14/);
    await user.selectOptions(screen.getByLabelText("Go to week"), "3");
    await user.click(screen.getByRole("button", { name: "Preview the move" }));

    expect(posts[0]).toEqual({ kind: "to_week", week: 3, dry_run: true });
    expect(await screen.findByText("week 1 → week 3")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Move the calendar" }));
    expect(await screen.findByText("Calendar moved. This week is week 3.")).toBeInTheDocument();
    expect(posts[1]).toEqual({ kind: "to_week", week: 3, dry_run: false });
  });

  it("cancelling a preview moves nothing", async () => {
    const posts = serve();
    const user = userEvent.setup();
    render(<CalendarPanel />, { wrapper: queryWrapper() });

    await screen.findByText(/This week is week 1 of 14/);
    await user.selectOptions(screen.getByLabelText("Go to week"), "5");
    await user.click(screen.getByRole("button", { name: "Preview the move" }));
    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(posts.map((body) => body.dry_run)).toEqual([true]);
    expect(screen.queryByRole("button", { name: "Move the calendar" })).toBeNull();
  });

  it("disables the match button when this week already is the furthest exercise's week", async () => {
    serve({ current: 1, anchor: 1, offset: 0 });
    render(<CalendarPanel />, { wrapper: queryWrapper() });
    expect(
      await screen.findByRole("button", { name: "Match my exercises (week 1)" }),
    ).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Back to the original dates" })).toBeNull();
  });

  it("offers the original dates only when the calendar has moved", async () => {
    serve({ current: 4, anchor: 2, offset: 3 });
    render(<CalendarPanel />, { wrapper: queryWrapper() });
    expect(
      await screen.findByRole("button", { name: "Back to the original dates" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Match my exercises (week 2)" })).toBeEnabled();
    expect(screen.getByText("Your furthest exercise is in week 2.")).toBeInTheDocument();
  });
});

describe("StartOverPanel", () => {
  it("keeps the confirm button disabled until the field reads reset, then shows the backup", async () => {
    const posts = serve();
    const user = userEvent.setup();
    render(<StartOverPanel />, { wrapper: queryWrapper() });

    await user.click(screen.getByRole("button", { name: "Start over" }));
    const confirm = screen.getByRole("button", { name: "Clear progress and start over" });
    expect(confirm).toBeDisabled();

    const field = screen.getByLabelText("Type reset to start over");
    await user.type(field, "rese");
    expect(confirm).toBeDisabled();
    await user.type(field, "t");
    expect(confirm).toBeEnabled();

    await user.click(confirm);
    expect(posts).toEqual([{ confirm: "reset" }]);
    expect(
      await screen.findByText("/scratch/var/backups/dogma-backup-20261103.json"),
    ).toBeInTheDocument();
    expect(screen.getByText("/scratch/var/backups/dogma-snapshot-20261103.db")).toBeInTheDocument();
  });

  it("cancelling the dialog sends nothing", async () => {
    const posts = serve();
    const user = userEvent.setup();
    render(<StartOverPanel />, { wrapper: queryWrapper() });
    await user.click(screen.getByRole("button", { name: "Start over" }));
    await user.type(screen.getByLabelText("Type reset to start over"), "reset");
    await user.click(screen.getByRole("button", { name: "Keep my progress" }));
    expect(posts).toEqual([]);
    expect(screen.queryByLabelText("Type reset to start over")).toBeNull();
  });
});

describe("CalendarOffer on Today", () => {
  function renderOffer() {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <CalendarOffer />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    return client;
  }

  it("offers the calendar at a two-week gap, linking to the Progress calendar", async () => {
    serve({ current: 3, anchor: 1, offset: 0 });
    const client = renderOffer();
    await waitFor(() => expect(client.getQueryState(["calendar"])?.status).toBe("success"));
    const link = await screen.findByRole("link", {
      name: "Move the calendar so this week is week 1?",
    });
    expect(link).toHaveAttribute("href", "/progress#calendar");
    expect(screen.getByText(/You haven't run an exercise yet\./)).toBeInTheDocument();
    expect(screen.queryByText(/Your furthest exercise/)).toBeNull();
  });

  it("names the furthest exercise's week once one has been run", async () => {
    serve({ current: 4, anchor: 2, offset: 0 });
    const client = renderOffer();
    await waitFor(() => expect(client.getQueryState(["calendar"])?.status).toBe("success"));
    expect(
      await screen.findByRole("link", { name: "Move the calendar so this week is week 2?" }),
    ).toHaveAttribute("href", "/progress#calendar");
    expect(screen.getByText(/Your furthest exercise is in week 2\./)).toBeInTheDocument();
  });

  it("stays quiet at a one-week gap", async () => {
    serve({ current: 2, anchor: 1, offset: 0 });
    const client = renderOffer();
    await waitFor(() => expect(client.getQueryState(["calendar"])?.status).toBe("success"));
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.queryByText(/Your furthest exercise/)).toBeNull();
  });
});

describe("calendar wording", () => {
  it("writes dates the way dogma warp does", () => {
    expect(dayLabel("2026-11-09")).toBe("Mon 09 Nov");
    expect(dayLabel("2027-01-31", true)).toBe("Sun 31 Jan 2027");
  });
});
