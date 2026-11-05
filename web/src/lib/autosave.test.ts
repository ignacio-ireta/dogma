import { afterEach, describe, expect, it, vi } from "vitest";
import { Autosave, SAVE_DELAY_MS, type SaveStatus, WORKSPACE_LIMIT_BYTES } from "./autosave";

function deferred() {
  let resolve = () => {};
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

afterEach(() => vi.useRealTimers());

describe("workspace autosave", () => {
  it("writes only the latest buffer after 800 ms of inactivity", async () => {
    vi.useFakeTimers();
    const writes: string[] = [];
    const autosave = new Autosave({
      save: async (text) => {
        writes.push(text);
      },
      onStatus: () => {},
    });
    autosave.edit("first");
    await vi.advanceTimersByTimeAsync(700);
    autosave.edit("second");
    await vi.advanceTimersByTimeAsync(SAVE_DELAY_MS - 1);
    expect(writes).toEqual([]);
    await vi.advanceTimersByTimeAsync(1);
    expect(writes).toEqual(["second"]);
  });

  it("serializes edits during an active write and flush waits for the newest text", async () => {
    const first = deferred();
    const writes: string[] = [];
    const autosave = new Autosave({
      save: async (text) => {
        writes.push(text);
        if (text === "first") await first.promise;
      },
      onStatus: () => {},
    });
    autosave.edit("first");
    const saved = autosave.flush();
    await Promise.resolve();
    autosave.edit("second");
    expect(writes).toEqual(["first"]);
    first.resolve();
    expect(await saved).toBe(true);
    expect(writes).toEqual(["first", "second"]);
  });

  it("refuses to flush an oversize UTF-8 buffer rather than grading stale disk content", async () => {
    const save = vi.fn(async () => {});
    const statuses: SaveStatus[] = [];
    const autosave = new Autosave({ save, onStatus: (status) => statuses.push(status) });
    autosave.reset("old disk file");
    autosave.edit("é".repeat(WORKSPACE_LIMIT_BYTES / 2 + 1));
    expect(await autosave.flush()).toBe(false);
    expect(save).not.toHaveBeenCalled();
    expect(statuses.at(-1)?.kind).toBe("too-large");
  });

  it("keeps the oversize status when an older in-flight save finishes", async () => {
    const first = deferred();
    const statuses: SaveStatus[] = [];
    const autosave = new Autosave({
      save: () => first.promise,
      onStatus: (status) => statuses.push(status),
      limitBytes: 4,
    });
    autosave.edit("old");
    const saved = autosave.flush();
    await Promise.resolve();
    autosave.edit("ééé");
    first.resolve();
    expect(await saved).toBe(false);
    expect(statuses.at(-1)?.kind).toBe("too-large");
    autosave.edit("new");
    expect(await autosave.flush()).toBe(true);
    expect(statuses.at(-1)?.kind).toBe("clean");
  });

  it("retries the latest edit after a failed save", async () => {
    const first = deferred();
    const writes: string[] = [];
    const statuses: SaveStatus[] = [];
    const autosave = new Autosave({
      save: async (text) => {
        writes.push(text);
        if (text === "first") {
          await first.promise;
          throw new Error("offline");
        }
      },
      onStatus: (status) => statuses.push(status),
    });
    autosave.edit("first");
    const saved = autosave.flush();
    await Promise.resolve();
    autosave.edit("second");
    first.resolve();
    expect(await saved).toBe(false);
    expect(statuses.at(-1)?.kind).toBe("error");
    expect(await autosave.flush()).toBe(true);
    expect(writes).toEqual(["first", "second"]);
  });
});

it("keeps the oversize warning if an older write fails", async () => {
  const first = deferred();
  const statuses: SaveStatus[] = [];
  const autosave = new Autosave({
    save: async () => {
      await first.promise;
      throw new Error("write failed");
    },
    onStatus: (status) => statuses.push(status),
    limitBytes: 4,
  });
  autosave.edit("old");
  const saved = autosave.flush();
  await Promise.resolve();
  autosave.edit("ééé");
  first.resolve();
  expect(await saved).toBe(false);
  expect(statuses.at(-1)?.kind).toBe("too-large");
  expect(await autosave.flush()).toBe(false);
});

it("accepts exactly 256 KiB of UTF-8 content", async () => {
  let writtenBytes = 0;
  const autosave = new Autosave({
    save: async (text) => {
      writtenBytes = new TextEncoder().encode(text).length;
    },
    onStatus: () => {},
  });
  autosave.edit("é".repeat(WORKSPACE_LIMIT_BYTES / 2));
  expect(await autosave.flush()).toBe(true);
  expect(writtenBytes).toBe(262144);
});
