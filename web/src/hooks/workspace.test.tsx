import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { WORKSPACE_LIMIT_BYTES } from "../lib/autosave";
import { attemptResult, deferred, jsonResponse, queryWrapper } from "../test/helpers";
import { useAttemptRun } from "./useAttemptRun";
import { useWorkspaceDraft } from "./useWorkspaceDraft";

afterEach(() => vi.unstubAllGlobals());
const id = "cml/topic/problem";

it.each(["starter text", "", null])(
  "creates a missing workspace once, even for an empty starter (%s)",
  async (starter) => {
    const writes: string[] = [];
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
      if (init.method === "PUT") {
        const body = JSON.parse(String(init.body)) as { content: string };
        writes.push(body.content);
        return jsonResponse({ problem_id: id, exists: true, content: body.content });
      }
      return jsonResponse({ problem_id: id, exists: false, content: null });
    });
    const { result, rerender, unmount } = renderHook(() => useWorkspaceDraft(id, starter), {
      wrapper: queryWrapper(),
    });
    await waitFor(() => expect(result.current.loaded).toBe(true));
    await waitFor(() => expect(writes).toEqual([starter ?? ""]));
    rerender();
    await act(async () => {
      await result.current.saveNow();
    });
    expect(writes).toHaveLength(1);
    unmount();
  },
);

it.each(["run", "submit"] as const)(
  "%s waits for all edits to save before sending an attempt",
  async (mode) => {
    const pendingSave = deferred<Response>();
    const requests: string[] = [];
    let first = true;
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      if (init.method === "PUT") {
        const body = JSON.parse(String(init.body)) as { content: string };
        requests.push(`save:${body.content}`);
        if (first) {
          first = false;
          return pendingSave.promise;
        }
        return jsonResponse({ problem_id: id, exists: true, content: body.content });
      }
      if (url.endsWith("/attempts")) {
        requests.push(`grade:${String(init.body)}`);
        return jsonResponse(attemptResult);
      }
      return jsonResponse({ problem_id: id, exists: true, content: "disk" });
    });
    const { result, unmount } = renderHook(
      () => {
        const draft = useWorkspaceDraft(id, "starter");
        const attempt = useAttemptRun(id, draft.saveNow);
        return { draft, attempt };
      },
      { wrapper: queryWrapper() },
    );
    await waitFor(() => expect(result.current.draft.loaded).toBe(true));
    act(() => result.current.draft.onChange("first edit"));
    let run: Promise<unknown> = Promise.resolve();
    act(() => {
      run = result.current.attempt.mutateAsync(mode);
    });
    await waitFor(() => expect(requests).toEqual(["save:first edit"]));
    act(() => result.current.draft.onChange("newest edit"));
    await act(async () => {
      pendingSave.resolve(jsonResponse({ problem_id: id, exists: true, content: "first edit" }));
      await run;
    });
    expect(requests).toEqual([
      "save:first edit",
      "save:newest edit",
      `grade:{"problem_id":"${id}","mode":"${mode}"}`,
    ]);
    unmount();
  },
);

it("never sends an attempt when the UTF-8 buffer exceeds the file limit", async () => {
  const requests: string[] = [];
  vi.stubGlobal("fetch", async (url: string) => {
    requests.push(url);
    return jsonResponse({ problem_id: id, exists: true, content: "old disk" });
  });
  const { result, unmount } = renderHook(
    () => {
      const draft = useWorkspaceDraft(id, "starter");
      return { draft, attempt: useAttemptRun(id, draft.saveNow) };
    },
    { wrapper: queryWrapper() },
  );
  await waitFor(() => expect(result.current.draft.loaded).toBe(true));
  act(() => result.current.draft.onChange("é".repeat(WORKSPACE_LIMIT_BYTES / 2 + 1)));
  await act(async () => {
    await expect(result.current.attempt.mutateAsync("submit")).rejects.toThrow("not graded");
  });
  expect(requests).toEqual([`/api/v1/workspace/${id}`]);
  expect(result.current.draft.status.kind).toBe("too-large");
  act(() => result.current.draft.onChange("small corrected file"));
  await act(async () => {
    await result.current.draft.saveNow();
  });
  unmount();
});

it("flushes late edits to their original problem on navigation", async () => {
  const writes: string[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    if (init.method === "PUT") {
      writes.push(`${url}:${String(init.body)}`);
    }
    return jsonResponse({ problem_id: id, exists: true, content: "disk" });
  });
  const { result, unmount } = renderHook(() => useWorkspaceDraft(id, "starter"), {
    wrapper: queryWrapper(),
  });
  await waitFor(() => expect(result.current.loaded).toBe(true));
  act(() => result.current.onChange("last edit"));
  unmount();
  await waitFor(() => expect(writes).toEqual([`/api/v1/workspace/${id}:{"content":"last edit"}`]));
});

it("waits for a departure save before loading the same problem again", async () => {
  const pending = deferred<Response>();
  let disk = "old disk";
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "PUT") {
      const response = await pending.promise;
      disk = "departure edit";
      return response;
    }
    return jsonResponse({ problem_id: id, exists: true, content: disk });
  });
  const wrapper = queryWrapper();
  const first = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await waitFor(() => expect(first.result.current.loaded).toBe(true));
  act(() => first.result.current.onChange("departure edit"));
  first.unmount();
  const second = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await act(async () => {
    await Promise.resolve();
  });
  expect(second.result.current.loaded).toBe(false);
  await act(async () => {
    pending.resolve(jsonResponse({ problem_id: id, exists: true, content: "departure edit" }));
  });
  await waitFor(() => expect(second.result.current.text).toBe("departure edit"));
  expect(second.result.current.loaded).toBe(true);
  second.unmount();
});

it("recovers unsaved text and its error after a failed departure save", async () => {
  const departure = deferred<Response>();
  let writes = 0;
  let disk = "old disk";
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "PUT") {
      writes += 1;
      if (writes === 1) return departure.promise;
      disk = (JSON.parse(String(init.body)) as { content: string }).content;
    }
    return jsonResponse({ problem_id: id, exists: true, content: disk });
  });
  const wrapper = queryWrapper();
  const first = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await waitFor(() => expect(first.result.current.loaded).toBe(true));
  act(() => first.result.current.onChange("irreplaceable unsaved edit"));
  first.unmount();
  await waitFor(() => expect(writes).toBe(1));
  await act(async () => departure.reject(new TypeError("network failed")));
  const second = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await waitFor(() => expect(second.result.current.loaded).toBe(true));
  expect(second.result.current.text).toBe("irreplaceable unsaved edit");
  expect(second.result.current.status).toMatchObject({
    kind: "error",
    message: expect.stringContaining("not reachable"),
  });
  await act(async () => expect(await second.result.current.saveNow()).toBe(true));
  expect(disk).toBe("irreplaceable unsaved edit");
  expect(second.result.current.status.kind).toBe("clean");
  second.unmount();
  const third = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await waitFor(() => expect(third.result.current.loaded).toBe(true));
  expect(third.result.current.text).toBe("irreplaceable unsaved edit");
  expect(third.result.current.status.kind).toBe("clean");
  third.unmount();
});

it("warns before browser unload while dirty, saving, or failed, including after navigation", async () => {
  const departure = deferred<Response>();
  let failing = true;
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "PUT" && failing) return departure.promise;
    return jsonResponse({ problem_id: id, exists: true, content: "disk" });
  });
  const warned = () => {
    const event = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(event);
    return event.defaultPrevented;
  };
  const wrapper = queryWrapper();
  const first = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await waitFor(() => expect(first.result.current.loaded).toBe(true));
  expect(warned()).toBe(false);
  act(() => first.result.current.onChange("unsaved"));
  expect(warned()).toBe(true);
  first.unmount();
  expect(warned()).toBe(true);
  await act(async () => departure.reject(new TypeError("network failed")));
  expect(warned()).toBe(true);
  const second = renderHook(() => useWorkspaceDraft(id, "starter"), { wrapper });
  await waitFor(() => expect(second.result.current.loaded).toBe(true));
  expect(warned()).toBe(true);
  failing = false;
  await act(async () => {
    await second.result.current.saveNow();
  });
  expect(warned()).toBe(false);
  second.unmount();
});
