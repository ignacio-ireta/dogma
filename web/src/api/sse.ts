// Server-sent events over `fetch` (ADR-0017). The tutor streams use POST, so EventSource (GET
// only) cannot carry them. The parser follows the WHATWG event-stream rules: CRLF, LF and CR line
// endings; comment lines; multi-line data; an event ends at a blank line; an event that is still
// open when the stream ends is discarded.
import { API_PREFIX, ApiProblem, problemFrom, send } from "./client";

export interface SseEvent {
  readonly event: string;
  readonly data: string;
}

interface LineEnd {
  readonly index: number;
  readonly length: number;
}

export class SseParser {
  private pending = "";
  private eventType = "";
  private dataLines: string[] = [];

  constructor(private readonly onEvent: (event: SseEvent) => void) {}

  /** Feed the next piece of the stream. Partial lines wait for the rest. */
  feed(chunk: string): void {
    this.pending += chunk;
    let start = 0;
    for (let end = findLineEnd(this.pending, start); end !== null; ) {
      this.line(this.pending.slice(start, end.index));
      start = end.index + end.length;
      end = findLineEnd(this.pending, start);
    }
    this.pending = this.pending.slice(start);
  }

  /** The stream ended. An open event (no blank line after it) is dropped, as the spec requires. */
  end(): void {
    // A CR at the very end is a line ending on its own (WHATWG); anything after it is partial.
    if (this.pending.endsWith("\r")) {
      this.feed("\n");
    }
    this.pending = "";
    this.reset();
  }

  private line(line: string): void {
    if (line === "") {
      this.dispatch();
      return;
    }
    if (line.startsWith(":")) {
      return;
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "event") {
      this.eventType = value;
    } else if (field === "data") {
      this.dataLines.push(value);
    }
    // `id` and `retry` carry nothing the browser uses.
  }

  private dispatch(): void {
    const dataLines = this.dataLines;
    const event = this.eventType || "message";
    this.reset();
    // An event with no data field at all is not dispatched. An empty `data:` field still is.
    if (dataLines.length === 0) {
      return;
    }
    this.onEvent({ event, data: dataLines.join("\n") });
  }

  private reset(): void {
    this.eventType = "";
    this.dataLines = [];
  }
}

/** The first line terminator at or after `from`. A trailing CR waits for the next chunk. */
function findLineEnd(text: string, from: number): LineEnd | null {
  for (let i = from; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "\n") {
      return { index: i, length: 1 };
    }
    if (ch === "\r") {
      if (i + 1 >= text.length) {
        return null;
      }
      return { index: i, length: text[i + 1] === "\n" ? 2 : 1 };
    }
  }
  return null;
}

export interface StreamOptions {
  readonly signal?: AbortSignal;
  readonly onEvent: (event: SseEvent) => void;
}

/**
 * POST a JSON body and feed each server-sent event to `onEvent` as it arrives. Resolves when the
 * stream ends (the caller decides whether a terminal event came). Refusals before the stream
 * starts are JSON problems and reject with ApiProblem. An abort resolves quietly.
 */
export async function postEventStream(
  path: string,
  body: unknown,
  options: StreamOptions,
): Promise<void> {
  const response = await send(`${API_PREFIX}${path}`, {
    method: "POST",
    headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: options.signal,
  }).catch((error: unknown) => {
    if (options.signal?.aborted) {
      return null;
    }
    throw error;
  });
  if (response === null) {
    return;
  }
  if (!response.ok) {
    throw await problemFrom(response);
  }
  if (response.body === null) {
    throw new ApiProblem({
      type: "about:blank",
      title: "No stream",
      status: response.status,
      detail: "The local API did not start a stream.",
      instance: path,
      code: "API_STREAM_MISSING",
      errors: null,
    });
  }
  const parser = new SseParser(options.onEvent);
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (let next = await reader.read(); !next.done; next = await reader.read()) {
      parser.feed(next.value);
    }
    parser.end();
  } catch (error) {
    if (options.signal?.aborted) {
      return;
    }
    throw error;
  }
}
