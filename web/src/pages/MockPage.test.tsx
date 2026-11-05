import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import type {
  MockDebriefOut,
  MockPlanOut,
  MockReportOut,
  MockStatusOut,
  RevealOut,
  RubricOut,
} from "../api/types";
import { jsonResponse, queryWrapper } from "../test/helpers";
import { MockPage } from "./MockPage";

const SESSION = "b".repeat(32);
const anchors = [
  { level: 1, text: "Did not demonstrate it." },
  { level: 2, text: "Attempted it, but weakly." },
  { level: 3, text: "Solid." },
  { level: 4, text: "Exceptional." },
];
const items = [
  { id: "coding.narration", text: "Explained the approach while writing the code." },
  { id: "coding.correctness", text: "The output is right for the stated task." },
  { id: "structure.approach", text: "Laid out the approach in a clear order." },
];
const rubric: RubricOut = {
  id: "mock",
  version: 1,
  kind: "mock",
  title: "Mock interview scoring",
  anchors,
  ready_score: null,
  maximum: 24,
  criteria: items.map((item) => item.id),
  dimensions: [],
  parts: [
    { id: "coding", title: "Live coding", minutes: "10-15", maximum: 8, items: items.slice(0, 2) },
    { id: "structure", title: "Case, structure", minutes: "12", maximum: 4, items: items.slice(2) },
  ],
  bands: [
    { label: "Developing", low: 0, high: 10, text: "Keep drilling the shape." },
    { label: "Solid", low: 11, high: 24, text: "A real interviewer would move on." },
  ],
};
const plan: MockPlanOut = {
  session_id: SESSION,
  started_at: new Date().toISOString(),
  case: { id: "sys/churn-prediction-case", title: "Churn prediction", brief: "Brief text here." },
  coding: null,
  theory: null,
  parts: [
    { title: "Live coding", minutes: "10-15" },
    { title: "Case, structure", minutes: "12" },
  ],
  focus: null,
};
const report: MockReportOut = {
  session_id: SESSION,
  parts: [
    {
      part_id: "coding",
      title: "Live coding",
      minutes: "10-15",
      score: 4,
      maximum: 8,
      percent: 50,
    },
    {
      part_id: "structure",
      title: "Case, structure",
      minutes: "12",
      score: 4,
      maximum: 4,
      percent: 100,
    },
  ],
  total: 8,
  maximum: 12,
  band: { label: "Developing", low: 0, high: 10, text: "Keep drilling the shape." },
  strongest: {
    part_id: "structure",
    title: "Case, structure",
    minutes: "12",
    score: 4,
    maximum: 4,
    percent: 100,
  },
  weakest: {
    part_id: "coding",
    title: "Live coding",
    minutes: "10-15",
    score: 4,
    maximum: 8,
    percent: 50,
  },
};
const debrief: MockDebriefOut = {
  report,
  case_id: "sys/churn-prediction-case",
  case_debrief: "A strong answer starts with the business question.",
};

function route(state: { open: MockPlanOut | null }) {
  const scored: Record<string, number>[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    const method = init.method ?? "GET";
    if (url.endsWith("/rubrics/mock")) return jsonResponse(rubric);
    if (url.endsWith("/mocks/current") && method === "GET") {
      const body: MockStatusOut = {
        open: state.open,
        last_scored: state.open === null && scored.length > 0 ? report : null,
      };
      return jsonResponse(body);
    }
    if (url.endsWith("/mocks") && method === "POST") {
      state.open = plan;
      return jsonResponse(plan, 201);
    }
    if (url.endsWith("/mocks/current/reveal")) {
      const reply: RevealOut = {
        text: "Churn is about 4 percent a month.",
        matched: true,
        exhibit: null,
      };
      return jsonResponse(reply);
    }
    if (url.endsWith("/mocks/current/score")) {
      scored.push(JSON.parse(String(init.body)).scores as Record<string, number>);
      state.open = null;
      return jsonResponse(report, 201);
    }
    if (url.endsWith(`/mocks/${SESSION}/debrief`)) return jsonResponse(debrief);
    return jsonResponse({ code: "API_NOT_FOUND" }, 404);
  });
  return scored;
}

function renderPage() {
  const Wrapper = queryWrapper();
  return render(
    <Wrapper>
      <MemoryRouter>
        <MockPage />
      </MemoryRouter>
    </Wrapper>,
  );
}

afterEach(() => vi.unstubAllGlobals());

it("walks start, reveal, score and debrief", async () => {
  const state: { open: MockPlanOut | null } = { open: null };
  const scored = route(state);
  renderPage();
  fireEvent.click(await screen.findByRole("button", { name: "Start a mock" }));

  expect(await screen.findByRole("heading", { name: "The case" })).toBeVisible();
  expect(screen.getByText("Brief text here.")).toBeVisible();
  expect(screen.getByRole("heading", { name: "Planned timings" })).toBeVisible();
  expect(screen.queryByText(/\b(overdue|late|red|hurry)\b/i)).toBeNull();

  fireEvent.change(screen.getByRole("textbox", { name: "Your question" }), {
    target: { value: "What is churn per month?" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
  expect(await screen.findByText("Churn is about 4 percent a month.")).toBeVisible();

  const save = screen.getByRole("button", { name: "Score the mock" });
  expect(save).toBeDisabled();
  expect(screen.getByText("3 items left to score.")).toBeVisible();
  for (const item of items) {
    const radio = document.querySelector<HTMLInputElement>(`input[name="${item.id}"][value="3"]`);
    expect(radio).not.toBeNull();
    fireEvent.click(radio as HTMLInputElement);
  }
  await waitFor(() => expect(save).toBeEnabled());
  fireEvent.click(save);

  expect(await screen.findByRole("heading", { name: "Debrief" })).toBeVisible();
  expect(screen.getByText("8 of 12")).toBeVisible();
  expect(screen.getByText("Live coding, 4 of 8 (50%)")).toBeVisible();
  expect(screen.getByText("Case, structure, 4 of 4 (100%)")).toBeVisible();
  expect(screen.getByText("A strong answer starts with the business question.")).toBeVisible();
  expect(scored).toHaveLength(1);
  expect(scored[0]?.["coding.narration"]).toBe(3);
});

it("shows the scale anchors once and keeps the score form grouped by part", async () => {
  route({ open: plan });
  renderPage();
  const scale = await screen.findByRole("list", { name: "What each score means" });
  expect(within(scale).getByText("Exceptional.")).toBeVisible();
  expect(screen.getByRole("group", { name: /Live coding/ })).toBeVisible();
  expect(screen.getByRole("group", { name: /Case, structure/ })).toBeVisible();
});
