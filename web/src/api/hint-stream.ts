// The hint stream's protocol (ROADMAP I-23): `reviewed` first, then `nudge` deltas, then exactly one
// terminal `done`, `unavailable` or `error`. Parsing and the display rules are pure, so every event
// order is covered by tests without rendering anything.
import { type ProblemDetails, problemFromError } from "./client";
import type { SseEvent } from "./sse";
import type { HintOut, ReviewedEventOut, ReviewedHintOut, UnavailableOut } from "./types";

export type HintEvent =
  | { readonly type: "reviewed"; readonly payload: ReviewedEventOut }
  | { readonly type: "nudge"; readonly text: string }
  | { readonly type: "done"; readonly payload: HintOut }
  | { readonly type: "unavailable"; readonly payload: UnavailableOut }
  | { readonly type: "error"; readonly payload: ProblemDetails };

export interface HintRequest {
  readonly problem_id: string;
  readonly level: number;
  readonly confirm_walkthrough: boolean;
}

export const STREAM_ENDED_PROBLEM: ProblemDetails = {
  type: "about:blank",
  title: "Stream ended early",
  status: 0,
  detail: "The hint stopped before it finished. Ask again.",
  instance: "",
  code: "API_STREAM_INCOMPLETE",
  errors: null,
};

/** Narrow one SSE event to a hint event. Unknown or malformed events are ignored (null). */
export function parseHintEvent(event: SseEvent): HintEvent | null {
  const body = parseJson(event.data);
  if (body === undefined) {
    return null;
  }
  switch (event.event) {
    case "reviewed":
      return isRecord(body) &&
        typeof body.level === "number" &&
        isReviewed(body.reviewed) &&
        isOptionalText(body.hints_note)
        ? { type: "reviewed", payload: body as ReviewedEventOut }
        : null;
    case "nudge":
      return isRecord(body) && typeof body.text === "string"
        ? { type: "nudge", text: body.text }
        : null;
    case "done":
      return isRecord(body) &&
        typeof body.level === "number" &&
        isReviewed(body.reviewed) &&
        (body.nudge === null || (isRecord(body.nudge) && typeof body.nudge.text === "string")) &&
        isOptionalText(body.nudge_note)
        ? { type: "done", payload: body as HintOut }
        : null;
    case "unavailable":
      return isRecord(body) && typeof body.code === "string" && typeof body.note === "string"
        ? { type: "unavailable", payload: body as UnavailableOut }
        : null;
    case "error":
      return isRecord(body) &&
        typeof body.code === "string" &&
        typeof body.detail === "string" &&
        typeof body.status === "number"
        ? { type: "error", payload: body as ProblemDetails }
        : null;
    default:
      return null;
  }
}

export type HintPhase = "idle" | "streaming" | "done" | "unavailable" | "error";

export interface HintState {
  readonly phase: HintPhase;
  /** The level being shown (from `reviewed`, or the level requested). */
  readonly level: number | null;
  readonly reviewed: ReviewedHintOut | null;
  readonly hintsNote: string | null;
  /** The model's raw text while it arrives. Not guarded: it is replaced by `nudge` on `done`. */
  readonly streamed: string;
  /** The guarded nudge from `done`, or null when the tutor gave none. */
  readonly nudge: string | null;
  readonly nudgeNote: string | null;
  readonly unavailableNote: string | null;
  readonly error: ProblemDetails | null;
}

export const initialHintState: HintState = {
  phase: "idle",
  level: null,
  reviewed: null,
  hintsNote: null,
  streamed: "",
  nudge: null,
  nudgeNote: null,
  unavailableNote: null,
  error: null,
};

export type HintAction =
  | { readonly type: "start"; readonly level: number }
  | { readonly type: "event"; readonly event: HintEvent }
  | { readonly type: "ended" }
  | { readonly type: "failed"; readonly error: unknown };

export function hintReducer(state: HintState, action: HintAction): HintState {
  switch (action.type) {
    case "start":
      return { ...initialHintState, phase: "streaming", level: action.level };
    case "failed":
      if (state.phase !== "streaming") return state;
      return {
        ...state,
        phase: "error",
        streamed: "",
        error: problemFromError(action.error),
      };
    case "ended":
      // Only a stream that is still open is incomplete; a finished one already has its outcome.
      return state.phase === "streaming"
        ? { ...state, phase: "error", streamed: "", error: STREAM_ENDED_PROBLEM }
        : state;
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: HintState, event: HintEvent): HintState {
  if (state.phase !== "streaming") {
    // The outcome is decided: anything after the terminal event is ignored.
    return state;
  }
  switch (event.type) {
    case "reviewed":
      return {
        ...state,
        level: event.payload.level,
        reviewed: event.payload.reviewed,
        hintsNote: event.payload.hints_note,
      };
    case "nudge":
      return { ...state, streamed: state.streamed + event.text };
    case "done":
      return {
        ...state,
        phase: "done",
        level: event.payload.level,
        reviewed: event.payload.reviewed ?? state.reviewed,
        streamed: "",
        nudge: event.payload.nudge?.text ?? null,
        nudgeNote: event.payload.nudge_note,
      };
    case "unavailable":
      return {
        ...state,
        phase: "unavailable",
        streamed: "",
        unavailableNote: event.payload.note,
      };
    case "error":
      return { ...state, phase: "error", streamed: "", error: event.payload };
  }
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isOptionalText(value: unknown): boolean {
  return value === null || typeof value === "string";
}

function isReviewed(value: unknown): boolean {
  return (
    value === null ||
    (isRecord(value) &&
      typeof value.level === "number" &&
      typeof value.label === "string" &&
      typeof value.text === "string")
  );
}
