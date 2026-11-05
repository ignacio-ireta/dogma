import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { deferred, jsonResponse, queryWrapper } from "../test/helpers";
import { MapPage } from "./MapPage";
import { ProgressPage } from "./ProgressPage";

type Schemas = components["schemas"];
const rows: Schemas["CoverageRowOut"][] = [
  {
    id: "cml/linear",
    track: "cml",
    tier: "A",
    title: "Linear models",
    kind: "topic",
    has_package: true,
    has_lesson: true,
    has_cards: true,
    has_interview: false,
    n_problems: 2,
    status: "verified",
  },
  {
    id: "cml/missing",
    track: "cml",
    tier: "A",
    title: "Future topic",
    kind: "topic",
    has_package: false,
    has_lesson: false,
    has_cards: false,
    has_interview: false,
    n_problems: 0,
    status: "missing",
  },
  {
    id: "sql/joins",
    track: "sql",
    tier: "A",
    title: "Joins",
    kind: "topic",
    has_package: true,
    has_lesson: false,
    has_cards: false,
    has_interview: false,
    n_problems: 0,
    status: "draft",
  },
  {
    id: "sql/advanced",
    track: "sql",
    tier: "B",
    title: "Advanced SQL",
    kind: "topic",
    has_package: true,
    has_lesson: true,
    has_cards: false,
    has_interview: true,
    n_problems: 0,
    status: "reviewed",
  },
];
const summary = (key: string, title: string): Schemas["CoverageSummaryOut"] => ({
  key,
  title,
  topics: 2,
  with_package: 1,
  with_lesson: 1,
  with_cards: 1,
  with_interview: 0,
  with_problems: 1,
  cases: 0,
  with_cases: 0,
  statuses: { verified: 1, missing: 1 },
});
const progress: Schemas["ProgressOut"] = {
  course: { id: "core", title: "Core path", start: "2026-10-01", end: "2026-12-31" },
  summary: {
    solved_topics: 1,
    total_topics: 4,
    current_week: 2,
    week_start: "2026-10-08",
    attempts_this_week: 7,
  },
  readiness: {
    next_due: [],
    recent_results: [],
    interview_ready: { ready: 0, total: 0, weeks: [] },
  },
  weeks: [
    {
      week: 1,
      theme: "Foundations",
      starts: "2026-10-01",
      solved: 1,
      total: 1,
      topics: [
        {
          topic_id: "cml/linear",
          title: "Linear models",
          status: "verified",
          mastery: "consolidating",
          solved: true,
          attempts: 3,
          best: { passed: 4, total: 5, mode: "submit" },
          last_attempt_at: "2026-10-02T12:00:00Z",
        },
      ],
    },
    {
      week: 2,
      theme: "Queries",
      starts: "2026-10-08",
      solved: 0,
      total: 1,
      topics: [
        {
          topic_id: "sql/joins",
          title: "Joins",
          status: "draft",
          mastery: "unseen",
          solved: false,
          attempts: 0,
          best: null,
          last_attempt_at: null,
        },
      ],
    },
  ],
};
const catalog: Schemas["CatalogOut"] = {
  version: 1,
  generated_at: "2026-10-01T00:00:00Z",
  topics: [
    {
      id: "cml/linear",
      title: "Linear models",
      track: "cml",
      tier: "A",
      status: "verified",
      requires: [],
      problems: [
        {
          id: "cml/linear/p02-optional-fit",
          title: "Optional fit",
          difficulty: "easy",
          language: "python",
          minutes: 15,
          status: "verified",
        },
        {
          id: "cml/linear/p01-core-fit",
          title: "Fit a line",
          difficulty: "easy",
          language: "python",
          minutes: 15,
          status: "verified",
        },
      ],
    },
  ],
};
const failure = (detail: string) =>
  jsonResponse(
    {
      type: "about:blank",
      title: "Failed",
      status: 500,
      code: "API_UNKNOWN",
      detail,
      instance: "",
      errors: null,
    },
    500,
  );

function serve(
  options: {
    empty?: boolean;
    coverageError?: boolean;
    progressError?: boolean;
    catalogError?: boolean;
    noBest?: boolean;
    draftCore?: boolean;
    unreadable?: boolean;
  } = {},
) {
  vi.stubGlobal("fetch", async (path: string) => {
    const url = new URL(path, "http://localhost");
    if (url.pathname.endsWith("/coverage")) {
      if (options.coverageError) return failure("Coverage is unavailable");
      const tier = url.searchParams.get("tier");
      const track = url.searchParams.get("track");
      const body: Schemas["CoverageOut"] = {
        tier: tier as "A" | "B" | "C" | null,
        track,
        by_track: [summary("cml", "Classical ML"), summary("sql", "SQL")],
        by_tier: [],
        rows:
          !tier && !track
            ? null
            : options.empty
              ? []
              : rows.filter(
                  (row) => (!tier || row.tier === tier) && (!track || row.track === track),
                ),
        unreadable_topics: options.unreadable ? ["cml/broken"] : [],
      };
      return jsonResponse(body);
    }
    if (url.pathname.endsWith("/progress")) {
      if (options.progressError) return failure("Progress is unavailable");
      const week = url.searchParams.get("week");
      return jsonResponse({
        ...progress,
        weeks: options.empty
          ? []
          : progress.weeks
              .filter((item) => !week || item.week === Number(week))
              .map((item) => ({
                ...item,
                topics: item.topics.map((topic) =>
                  options.noBest ? { ...topic, best: null } : topic,
                ),
              })),
      });
    }
    if (url.pathname.endsWith("/catalog"))
      return options.catalogError
        ? failure("Core links are unavailable")
        : jsonResponse(
            options.draftCore
              ? {
                  ...catalog,
                  topics: catalog.topics.map((topic) => ({
                    ...topic,
                    problems: topic.problems.map((problem) => ({ ...problem, status: "draft" })),
                  })),
                }
              : catalog,
          );
    throw new Error(`Unexpected request: ${path}`);
  });
}
function page(component: "map" | "progress") {
  return render(
    <MemoryRouter>{component === "map" ? <MapPage /> : <ProgressPage />}</MemoryRouter>,
    { wrapper: queryWrapper() },
  );
}
afterEach(() => vi.unstubAllGlobals());

it("does not call recorded attempts absent when they have no usable test result", async () => {
  serve({ noBest: true });
  page("progress");
  const card = await screen.findByRole("article", { name: "Linear models" });
  expect(within(card).getByText("3 attempts")).toBeVisible();
  expect(within(card).getByText("No test result recorded.")).toBeVisible();
  expect(within(card).queryByText("No attempts recorded.")).not.toBeInTheDocument();
});

it("keeps draft core problems unavailable even when their topic is verified", async () => {
  serve({ draftCore: true });
  page("map");
  const card = await screen.findByRole("article", { name: "Linear models" });
  expect(await within(card).findByText("No verified core problem available.")).toBeVisible();
  expect(within(card).queryByRole("link")).not.toBeInTheDocument();
});

it("names unreadable packages without counting them as authored", async () => {
  serve({ unreadable: true });
  page("map");
  expect(
    await screen.findByText(/Some topic packages could not be read: cml\/broken/),
  ).toBeVisible();
});

it("defaults Map to tier A, groups tracks and applies tier and track filters", async () => {
  serve();
  page("map");
  expect(await screen.findByRole("heading", { name: "Classical ML" })).toBeVisible();
  expect(screen.getByLabelText("Tier")).toHaveValue("A");
  expect(screen.getByRole("heading", { name: "Joins" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Advanced SQL" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Track"), { target: { value: "cml" } });
  await screen.findByRole("heading", { name: "Linear models" });
  expect(screen.queryByRole("heading", { name: "Joins" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Track"), { target: { value: "" } });
  fireEvent.change(screen.getByLabelText("Tier"), { target: { value: "B" } });
  expect(await screen.findByRole("heading", { name: "Advanced SQL" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Linear models" })).not.toBeInTheDocument();
});

it("separates content status from recorded mastery and links only authored core problems", async () => {
  serve();
  page("map");
  const card = await screen.findByRole("article", { name: "Linear models" });
  expect(within(card).getByText("Content: Verified")).toBeVisible();
  expect(await within(card).findByText("Mastery: Consolidating")).toBeVisible();
  expect(await within(card).findByRole("link", { name: "Fit a line" })).toHaveAttribute(
    "href",
    "/problem/cml/linear/p01-core-fit",
  );
  expect(screen.queryByRole("link", { name: "Optional fit" })).not.toBeInTheDocument();
  const missing = screen.getByRole("article", { name: "Future topic" });
  expect(within(missing).getByText("Content: Missing")).toBeVisible();
  expect(within(missing).queryByRole("link")).not.toBeInTheDocument();
  expect(within(missing).getByText("Lesson: Missing")).toBeVisible();
  expect(within(missing).getByText("Mastery: Not recorded in core path")).toBeVisible();
  expect(
    within(screen.getByRole("article", { name: "Joins" })).getByText("Content: Draft"),
  ).toBeVisible();
});

it("does not turn an unavailable mastery request into unseen mastery", async () => {
  serve({ progressError: true });
  page("map");
  expect(await screen.findByText("Progress is unavailable")).toBeVisible();
  expect(screen.getAllByText("Mastery: Unavailable").length).toBeGreaterThan(0);
  expect(screen.queryByText("Mastery: Unseen")).not.toBeInTheDocument();
});

it("keeps coverage visible when core problem links fail to load", async () => {
  serve({ catalogError: true });
  page("map");
  expect(await screen.findByText("Core links are unavailable")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Linear models" })).toBeVisible();
  expect(screen.queryByRole("link", { name: "Fit a line" })).not.toBeInTheDocument();
  expect(screen.getAllByText("Core problem link unavailable.").length).toBeGreaterThan(0);
  expect(screen.queryByText("No verified core problem available.")).not.toBeInTheDocument();
});

it("shows Map loading before its response and an honest empty filter result", async () => {
  const response = deferred<Response>();
  vi.stubGlobal("fetch", () => response.promise);
  page("map");
  expect(screen.getByText("Loading the topic map…")).toBeVisible();
  await act(async () =>
    response.resolve(
      jsonResponse({
        tier: "A",
        track: null,
        by_track: [],
        by_tier: [],
        rows: [],
        unreadable_topics: [],
      }),
    ),
  );
  expect(await screen.findByText("No topics match these filters.")).toBeVisible();
});

it("shows the coverage error with a retry action", async () => {
  serve({ coverageError: true });
  page("map");
  expect(await screen.findByText("Coverage is unavailable")).toBeVisible();
  serve();
  fireEvent.click(screen.getByRole("button", { name: "Retry topic map" }));
  expect(await screen.findByRole("heading", { name: "Linear models" })).toBeVisible();
});

it("renders actual progress counts, week data, mastery and attempt results", async () => {
  serve();
  page("progress");
  expect(await screen.findByText("1 of 4 core topics solved")).toBeVisible();
  expect(screen.getByText("7 attempts this week")).toBeVisible();
  expect(screen.getByText("Current week: 2 · starts 2026-10-08")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Week 1 · Foundations" })).toBeVisible();
  expect(screen.getByText("3 attempts")).toBeVisible();
  expect(screen.getByText("Best result: 4/5 tests · submit")).toBeVisible();
  expect(screen.getByText("Mastery: Consolidating")).toBeVisible();
  expect(screen.getByText("No attempts recorded.")).toBeVisible();
  expect(screen.queryByText(/retention|streak|debt/i)).not.toBeInTheDocument();
});

it("requests the selected progress week and allows returning to every week", async () => {
  serve();
  page("progress");
  await screen.findByRole("heading", { name: "Week 1 · Foundations" });
  fireEvent.change(screen.getByLabelText("Study week"), { target: { value: "2" } });
  expect(await screen.findByRole("heading", { name: "Week 2 · Queries" })).toBeVisible();
  expect(screen.queryByRole("heading", { name: "Week 1 · Foundations" })).not.toBeInTheDocument();
  expect(screen.getByText("1 of 4 core topics solved")).toBeVisible();
  fireEvent.change(screen.getByLabelText("Study week"), { target: { value: "" } });
  expect(await screen.findByRole("heading", { name: "Week 1 · Foundations" })).toBeVisible();
});

it("shows an empty progress history without inventing outcomes", async () => {
  serve({ empty: true });
  page("progress");
  expect(await screen.findByText("No study weeks are available.")).toBeVisible();
  expect(screen.queryByText(/Best result/)).not.toBeInTheDocument();
});

it("shows Progress loading and a recoverable server error", async () => {
  const response = deferred<Response>();
  vi.stubGlobal("fetch", () => response.promise);
  page("progress");
  expect(screen.getByText("Loading your progress…")).toBeVisible();
  await act(async () => response.resolve(failure("Progress could not load")));
  expect(await screen.findByText("Progress could not load")).toBeVisible();
  serve();
  fireEvent.click(screen.getByRole("button", { name: "Retry progress" }));
  expect(await screen.findByText("1 of 4 core topics solved")).toBeVisible();
});
