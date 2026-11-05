import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { AnswerOut, AnswerRequest, InterviewOut, RubricOut } from "../../api/types";
import { jsonResponse, queryWrapper } from "../../test/helpers";
import { AnswerPanel } from "./AnswerPanel";

const topic = "cml/vector-operations";
const anchors = [
  { level: 1, text: "Wrong, vague or missing." },
  { level: 2, text: "Muddled." },
  { level: 3, text: "Right and clear." },
  { level: 4, text: "Precise and complete." },
];
const dimension = (id: string, title: string) => ({
  id,
  title,
  question: `Is it ${title.toLowerCase()}?`,
  anchors,
});
const rubric: RubricOut = {
  id: "topic-answer",
  version: 1,
  kind: "topic-answer",
  title: "Topic interview answer",
  anchors,
  ready_score: 3,
  maximum: 16,
  criteria: ["accuracy", "structure", "trade-offs", "communication"],
  dimensions: [
    dimension("accuracy", "Accuracy"),
    dimension("structure", "Structure"),
    dimension("trade-offs", "Trade-offs"),
    dimension("communication", "Communication"),
  ],
  parts: [],
  bands: [],
};
const interview: InterviewOut = {
  topic_id: topic,
  title: "Vector operations",
  mastery: "retained",
  questions: [{ number: 1, title: "What does the dot product measure, geometrically?" }],
  rubric,
};
const saved: AnswerOut = {
  answer_id: "a".repeat(32),
  topic_id: topic,
  prompt_ref: "interview.md#Q1",
  scores: { accuracy: 3, structure: 3, "trade-offs": 3, communication: 3 },
  spoken: false,
  before: "retained",
  after: "interview-ready",
  promoted: true,
  ready_at: "2026-10-28T09:00:00Z",
  reason: "Interview-ready: retained, and every dimension scored 3 or higher.",
};

function routeApi(post: (body: AnswerRequest) => AnswerOut) {
  const sent: AnswerRequest[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit = {}) => {
    if (url.endsWith("/interview")) return jsonResponse(interview);
    if (url.endsWith("/rubrics/topic-answer")) return jsonResponse(rubric);
    if (url.endsWith("/answers") && init.method === "POST") {
      const body = JSON.parse(String(init.body)) as AnswerRequest;
      sent.push(body);
      return jsonResponse(post(body), 201);
    }
    return jsonResponse({ code: "API_NOT_FOUND" }, 404);
  });
  return sent;
}

function renderPanel() {
  const Wrapper = queryWrapper();
  return render(
    <Wrapper>
      <AnswerPanel topicId={topic} />
    </Wrapper>,
  );
}

afterEach(() => vi.unstubAllGlobals());

it("shows every dimension with its anchors visible, before any score is given", async () => {
  routeApi(() => saved);
  renderPanel();
  expect(
    await screen.findByRole("heading", { name: "Answer an interview question" }),
  ).toBeVisible();
  for (const label of ["Accuracy", "Structure", "Trade-offs", "Communication"]) {
    expect(screen.getByRole("group", { name: new RegExp(label) })).toBeVisible();
  }
  expect(screen.getAllByText("Wrong, vague or missing.")).toHaveLength(4);
  expect(screen.getByRole("button", { name: "Save answer" })).toBeDisabled();
});

it("saves a written answer with the four scores and shows what it did", async () => {
  const sent = routeApi(() => saved);
  renderPanel();
  await screen.findByRole("heading", { name: "Answer an interview question" });
  fireEvent.change(screen.getByLabelText("Your answer"), {
    target: { value: "It measures how aligned two vectors are." },
  });
  for (const dimensionTitle of ["Accuracy", "Structure", "Trade-offs", "Communication"]) {
    const group = screen.getByRole("group", { name: new RegExp(dimensionTitle) });
    fireEvent.click(group.querySelectorAll("input")[2] as HTMLElement);
  }
  const save = screen.getByRole("button", { name: "Save answer" });
  await waitFor(() => expect(save).toBeEnabled());
  fireEvent.click(save);
  expect(await screen.findByText("This topic is now interview-ready.")).toBeVisible();
  expect(screen.getByText("retained → interview-ready")).toBeVisible();
  expect(sent).toHaveLength(1);
  expect(sent[0]).toEqual({
    question: 1,
    spoken: false,
    text: "It measures how aligned two vectors are.",
    scores: { accuracy: 3, structure: 3, "trade-offs": 3, communication: 3 },
  });
});

it("sends no text when the answer was given aloud", async () => {
  const sent = routeApi(() => ({ ...saved, spoken: true, promoted: false, after: "retained" }));
  renderPanel();
  await screen.findByRole("heading", { name: "Answer an interview question" });
  fireEvent.click(screen.getByLabelText(/I answered it aloud/));
  for (const dimensionTitle of ["Accuracy", "Structure", "Trade-offs", "Communication"]) {
    const group = screen.getByRole("group", { name: new RegExp(dimensionTitle) });
    fireEvent.click(group.querySelectorAll("input")[3] as HTMLElement);
  }
  fireEvent.click(screen.getByRole("button", { name: "Save answer" }));
  await screen.findByText("Answer saved");
  expect(sent[0]).not.toHaveProperty("text");
  expect(sent[0]?.spoken).toBe(true);
});
