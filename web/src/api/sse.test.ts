import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiProblem } from "./client";
import { postEventStream, type SseEvent, SseParser } from "./sse";

afterEach(() => vi.unstubAllGlobals());

describe("SSE framing", () => {
  it.each(["\n", "\r\n", "\r"])("handles every chunk split with %j line endings", (newline) => {
    const input = [
      ":comment",
      "event: reviewed",
      "data: first",
      "data: second",
      "",
      "data:",
      "",
      "",
      "event: ignored",
      "",
      "event: done",
      'data: {"level":1}',
      "",
      "",
    ].join(newline);
    for (let split = 0; split <= input.length; split += 1) {
      const events: SseEvent[] = [];
      const parser = new SseParser((event) => events.push(event));
      parser.feed(input.slice(0, split));
      parser.feed(input.slice(split));
      parser.end();
      expect(events).toEqual([
        { event: "reviewed", data: "first\nsecond" },
        { event: "message", data: "" },
        { event: "done", data: '{"level":1}' },
      ]);
    }
  });

  it("drops a terminal event without its final blank line", () => {
    const events: SseEvent[] = [];
    const parser = new SseParser((event) => events.push(event));
    parser.feed('event: reviewed\ndata: {}\n\nevent: done\ndata: {"level":1}\n');
    parser.end();
    expect(events).toEqual([{ event: "reviewed", data: "{}" }]);
  });

  it("decodes UTF-8 code points split across network chunks", async () => {
    const bytes = new TextEncoder().encode('event: nudge\ndata: {"text":"é🙂"}\n\n');
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
              controller.close();
            },
          }),
        ),
    );
    const events: SseEvent[] = [];
    await postEventStream("/hint", {}, { onEvent: (event) => events.push(event) });
    expect(events).toEqual([{ event: "nudge", data: '{"text":"é🙂"}' }]);
  });

  it("preserves an API problem when a POST is refused before streaming", async () => {
    vi.stubGlobal(
      "fetch",
      async () =>
        new Response(
          JSON.stringify({
            type: "about:blank",
            title: "Locked",
            status: 403,
            detail: "Submit twice first",
            code: "HINT_LOCKED",
            instance: "",
            errors: null,
          }),
          { status: 403 },
        ),
    );
    await expect(postEventStream("/hint", {}, { onEvent: () => {} })).rejects.toBeInstanceOf(
      ApiProblem,
    );
  });
});
