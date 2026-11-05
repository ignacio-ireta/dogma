import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { deferred } from "../test/helpers";
import { useHintStream } from "./useHintStream";

afterEach(() => vi.unstubAllGlobals());

it("ignores events from an aborted request that finishes after a newer hint starts", async () => {
  const old = deferred<Response>();
  let latest: ReadableStreamDefaultController<Uint8Array> | undefined;
  let count = 0;
  const signals: (AbortSignal | null | undefined)[] = [];
  vi.stubGlobal("fetch", (_url: string, init: RequestInit) => {
    signals.push(init.signal);
    count += 1;
    if (count === 1) return old.promise;
    return Promise.resolve(
      new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            latest = controller;
          },
        }),
      ),
    );
  });
  const { result, unmount } = renderHook(() => useHintStream("cml/topic/problem"));
  let first = Promise.resolve();
  let second = Promise.resolve();
  act(() => {
    first = result.current.request(1, false);
  });
  act(() => {
    second = result.current.request(2, false);
  });
  await waitFor(() => expect(result.current.state.level).toBe(2));
  expect(signals[0]?.aborted).toBe(true);
  await act(async () => {
    old.resolve(
      new Response(
        'event: reviewed\ndata: {"level":1,"reviewed":{"level":1,"label":"Old","text":"Old hint"},"hints_note":null}\n\nevent: done\ndata: {"level":1,"reviewed":null,"nudge":null,"nudge_note":null}\n\n',
      ),
    );
    await first;
  });
  expect(result.current.state).toMatchObject({ level: 2, phase: "streaming", reviewed: null });
  await act(async () => {
    latest?.enqueue(
      new TextEncoder().encode(
        'event: done\ndata: {"level":2,"reviewed":null,"nudge":null,"nudge_note":null}\n\n',
      ),
    );
    latest?.close();
    await second;
  });
  expect(result.current.state).toMatchObject({ level: 2, phase: "done" });
  unmount();
});

it("aborts an active hint when the problem page unmounts", async () => {
  let signal: AbortSignal | null | undefined;
  vi.stubGlobal("fetch", (_url: string, init: RequestInit) => {
    signal = init.signal;
    return new Promise(() => {});
  });
  const { result, unmount } = renderHook(() => useHintStream("cml/topic/problem"));
  act(() => {
    void result.current.request(1, false);
  });
  unmount();
  expect(signal?.aborted).toBe(true);
});
