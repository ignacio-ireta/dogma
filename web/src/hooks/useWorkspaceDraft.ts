// The learner's file for one problem: loaded from the API, seeded from the starter when missing,
// and saved with debounced autosave. Unsaved departure buffers stay in memory until recovered.
import { type QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getWorkspace, putWorkspace } from "../api/endpoints";
import { Autosave, type SaveStatus } from "../lib/autosave";

// A quick return to a problem must wait for the write started by its departing page.
// Failed writes retain their text and status for a return to the page. The API is durable truth;
// this recovery buffer lasts only for the current browser session and QueryClient.
interface DepartingDraft {
  readonly text: string;
  status: SaveStatus;
  readonly saving: Promise<boolean>;
  readonly warn: (event: BeforeUnloadEvent) => void;
}
const departingSaves = new WeakMap<QueryClient, Map<string, DepartingDraft>>();

function warnUnsaved(event: BeforeUnloadEvent): void {
  event.preventDefault();
  event.returnValue = "";
}

export interface WorkspaceDraft {
  readonly text: string;
  readonly loaded: boolean;
  readonly status: SaveStatus;
  readonly onChange: (next: string) => void;
  /** Save the buffer now. Resolves false when the buffer could not be saved. */
  readonly saveNow: () => Promise<boolean>;
  readonly loadError: unknown;
}

export function useWorkspaceDraft(
  problemId: string,
  starter: string | null | undefined,
): WorkspaceDraft {
  const queryClient = useQueryClient();
  let pending = departingSaves.get(queryClient);
  if (pending === undefined) {
    pending = new Map();
    departingSaves.set(queryClient, pending);
  }
  const query = useQuery({
    queryKey: ["workspace", problemId],
    queryFn: async () => {
      await pending.get(problemId)?.saving;
      return getWorkspace(problemId);
    },
    enabled: problemId !== "",
    retry: false,
  });
  const [text, setText] = useState("");
  const [status, setStatus] = useState<SaveStatus>({ kind: "clean" });
  const [loaded, setLoaded] = useState(false);
  const seeded = useRef<string | null>(null);
  const live = useRef<{ text: string; status: SaveStatus; loaded: boolean }>({
    text: "",
    status: { kind: "clean" },
    loaded: false,
  });
  const updateStatus = useCallback((next: SaveStatus) => {
    live.current.status = next;
    setStatus(next);
  }, []);

  const autosave = useMemo(
    () =>
      new Autosave({
        save: async (content) => {
          const saved = await putWorkspace(problemId, content);
          queryClient.setQueryData(["workspace", problemId], saved);
        },
        onStatus: updateStatus,
      }),
    [problemId, queryClient, updateStatus],
  );

  useEffect(() => {
    return () => {
      // An unfinished load must not replace a failed buffer from the previous page.
      if (!live.current.loaded) return;
      const entry: DepartingDraft = {
        text: live.current.text,
        status: live.current.status,
        saving: autosave.flush(),
        warn: (event) => warnUnsaved(event),
      };
      pending.set(problemId, entry);
      if (entry.status.kind !== "clean") window.addEventListener("beforeunload", entry.warn);
      void entry.saving.then((saved) => {
        entry.status = live.current.status;
        if (saved && pending.get(problemId) === entry) {
          pending.delete(problemId);
          window.removeEventListener("beforeunload", entry.warn);
        }
      });
    };
  }, [autosave, pending, problemId]);

  useEffect(() => {
    if (status.kind === "clean") return;
    const warn = (event: BeforeUnloadEvent) => warnUnsaved(event);
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [status.kind]);

  const data = query.data;
  const starterReady = starter !== undefined;
  const fetching = query.isFetching;
  useEffect(() => {
    if (fetching || data === undefined || !starterReady || seeded.current === problemId) {
      return;
    }
    seeded.current = problemId;
    const recovery = pending.get(problemId);
    if (recovery !== undefined) {
      pending.delete(problemId);
      window.removeEventListener("beforeunload", recovery.warn);
      live.current.text = recovery.text;
      setText(recovery.text);
      autosave.reset(data.exists ? (data.content ?? "") : null);
      autosave.edit(recovery.text);
      updateStatus(recovery.status);
    } else if (data.exists) {
      const content = data.content ?? "";
      live.current.text = content;
      setText(content);
      autosave.reset(content);
    } else {
      const initial = starter ?? "";
      live.current.text = initial;
      setText(initial);
      autosave.reset(null);
      autosave.edit(initial);
      void autosave.flush();
    }
    live.current.loaded = true;
    setLoaded(true);
  }, [data, starter, starterReady, problemId, autosave, fetching, pending, updateStatus]);

  const onChange = useCallback(
    (next: string) => {
      live.current.text = next;
      setText(next);
      autosave.edit(next);
    },
    [autosave],
  );

  const saveNow = useCallback(() => autosave.flush(), [autosave]);

  return { text, loaded, status, onChange, saveNow, loadError: query.error };
}
