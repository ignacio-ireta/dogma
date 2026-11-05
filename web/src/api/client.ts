// The fetch client for the local API (ARCHITECTURE §4.7). Same origin in production; the Vite
// dev server proxies /api during development. Errors are RFC 9457 problems with a domain code.
import type { components } from "./schema";

export type ProblemDetails = components["schemas"]["Problem"];

export const API_PREFIX = "/api/v1";

/** A problem the API answered with (4xx or 5xx, `application/problem+json`). */
export class ApiProblem extends Error {
  readonly problem: ProblemDetails;

  constructor(problem: ProblemDetails) {
    super(problem.detail);
    this.name = "ApiProblem";
    this.problem = problem;
  }
}

/** The local API did not answer at all (not running, or a dropped connection). */
export class ApiUnreachable extends Error {
  constructor() {
    super("The local API is not reachable. Start it with `dogma serve`.");
    this.name = "ApiUnreachable";
  }
}

/** A message the browser wrote itself, safe to show as it is (for example, a save that failed). */
export class LocalError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LocalError";
  }
}

/** Encode each segment of a problem or topic id but keep its slashes (`track/topic/problem`). */
export function encodeId(id: string): string {
  return id.split("/").map(encodeURIComponent).join("/");
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Accept", "application/json");
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await send(`${API_PREFIX}${path}`, { ...init, headers });
  if (!response.ok) {
    throw await problemFrom(response);
  }
  const body: unknown = await response.json();
  return body as T;
}

/** Send a request, turning a network failure into ApiUnreachable. */
export async function send(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (error) {
    if (init.signal?.aborted) {
      throw error;
    }
    throw new ApiUnreachable();
  }
}

/** The problem an error from the API client stands for, so every failure reads the same way. */
export function problemFrom(response: Response): Promise<ApiProblem> {
  return response
    .json()
    .then((body: unknown) => {
      if (isProblem(body)) {
        return new ApiProblem(body);
      }
      return new ApiProblem(fallbackProblem(response));
    })
    .catch(() => new ApiProblem(fallbackProblem(response)));
}

function fallbackProblem(response: Response): ProblemDetails {
  return {
    type: "about:blank",
    title: response.statusText || "Request failed",
    status: response.status,
    detail: `The local API answered with status ${response.status}.`,
    instance: "",
    code: "API_UNKNOWN",
    errors: null,
  };
}

function isProblem(value: unknown): value is ProblemDetails {
  return (
    typeof value === "object" &&
    value !== null &&
    "code" in value &&
    typeof value.code === "string" &&
    "detail" in value &&
    typeof value.detail === "string" &&
    "status" in value &&
    typeof value.status === "number"
  );
}

/** The sentence to show a learner for any failure. */
export function describeError(error: unknown): string {
  if (error instanceof ApiProblem) {
    return error.problem.detail;
  }
  if (error instanceof ApiUnreachable || error instanceof LocalError) {
    return error.message;
  }
  return "Something went wrong. Try again.";
}

/** A problem value for failures that never produced one (for event-stream errors). */
export function problemFromError(error: unknown): ProblemDetails {
  if (error instanceof ApiProblem) {
    return error.problem;
  }
  return {
    type: "about:blank",
    title: "Request failed",
    status: 0,
    detail: describeError(error),
    instance: "",
    code: "API_CLIENT_ERROR",
    errors: null,
  };
}
