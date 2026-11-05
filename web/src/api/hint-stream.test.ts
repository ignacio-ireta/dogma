import { describe, expect, it } from "vitest";
import {
  type HintEvent,
  type HintState,
  hintReducer,
  initialHintState,
  parseHintEvent,
} from "./hint-stream";

const reviewed = { level: 1, label: "Concept", text: "Authored hint" };
const problem = {
  type: "about:blank",
  title: "Error",
  status: 503,
  detail: "Try again",
  instance: "",
  code: "UNAVAILABLE",
  errors: null,
};
const start = () => hintReducer(initialHintState, { type: "start", level: 1 });
const event = (state: HintState, value: HintEvent) =>
  hintReducer(state, { type: "event", event: value });
const done: HintEvent = {
  type: "done",
  payload: { level: 1, reviewed: null, nudge: null, nudge_note: "No nudge" },
};

it("shows reviewed immediately and replaces raw streamed text with guarded final text", () => {
  let state = event(start(), {
    type: "reviewed",
    payload: { level: 1, reviewed, hints_note: null },
  });
  expect(state.reviewed?.text).toBe("Authored hint");
  state = event(state, { type: "nudge", text: "raw " });
  state = event(state, { type: "nudge", text: "draft" });
  expect(state.streamed).toBe("raw draft");
  state = event(state, done);
  expect(state).toMatchObject({
    phase: "done",
    streamed: "",
    reviewed,
    nudge: null,
    nudgeNote: "No nudge",
  });
});

it.each([
  done,
  { type: "unavailable", payload: { code: "MODEL_OFFLINE", note: "Start the model" } },
  { type: "error", payload: problem },
] satisfies HintEvent[])(
  "keeps reviewed through terminal $type and ignores later events",
  (terminal) => {
    let state = event(start(), {
      type: "reviewed",
      payload: { level: 1, reviewed, hints_note: "Authored" },
    });
    state = event(state, { type: "nudge", text: "unfinished" });
    state = event(state, terminal);
    expect(state.reviewed).toEqual(reviewed);
    expect(state.streamed).toBe("");
    expect(event(state, { type: "nudge", text: "late" })).toBe(state);
    expect(hintReducer(state, { type: "ended" })).toBe(state);
  },
);

it("does not replace a completed hint when the network breaks after done", () => {
  const state = event(start(), done);
  expect(hintReducer(state, { type: "failed", error: new Error("connection reset") })).toBe(state);
});

it("keeps the reviewed hint but discards unfinished generated text on truncated streams", () => {
  let state = event(start(), {
    type: "reviewed",
    payload: { level: 1, reviewed, hints_note: null },
  });
  state = event(state, { type: "nudge", text: "unfinished" });
  state = hintReducer(state, { type: "ended" });
  expect(state).toMatchObject({
    phase: "error",
    reviewed,
    streamed: "",
    error: { code: "API_STREAM_INCOMPLETE" },
  });
  expect(hintReducer(state, { type: "start", level: 2 })).toMatchObject({
    phase: "streaming",
    level: 2,
    reviewed: null,
    error: null,
  });
});

describe("hint event validation", () => {
  it.each([
    ["reviewed", { level: 1 }],
    ["reviewed", { level: 1, reviewed: {}, hints_note: null }],
    ["done", { level: 1 }],
    ["done", { level: 1, reviewed: null, nudge: {}, nudge_note: null }],
    ["error", { code: "ERROR" }],
  ])("ignores malformed %s payload %j", (name, body) => {
    expect(parseHintEvent({ event: String(name), data: JSON.stringify(body) })).toBeNull();
  });
  it("ignores unknown events and invalid JSON", () => {
    expect(parseHintEvent({ event: "unknown", data: "{}" })).toBeNull();
    expect(parseHintEvent({ event: "reviewed", data: "{" })).toBeNull();
  });
  it("accepts a complete reviewed event", () => {
    expect(
      parseHintEvent({
        event: "reviewed",
        data: JSON.stringify({ level: 1, reviewed, hints_note: null }),
      }),
    ).toEqual({ type: "reviewed", payload: { level: 1, reviewed, hints_note: null } });
  });
});
