import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { describeError } from "../../api/client";
import {
  type CloseOut,
  type CloseRequest,
  closeSession,
  type SessionOut,
  type TodayOut,
} from "../../api/learning";
import { Notice } from "../../components/Notice";

export const closeFields = [
  ["stopped_where", "Where I stopped"],
  ["next_step", "Exact next step"],
  ["blocker", "Blocker (optional)"],
  ["answer", "60-second answer (optional)"],
] as const;

export function CloseForm({
  session,
  disabled,
  onClosed,
  onPendingChange,
}: {
  readonly session: SessionOut;
  readonly disabled: boolean;
  readonly onClosed: (result: CloseOut, draft: CloseRequest) => void;
  readonly onPendingChange: (pending: boolean) => void;
}) {
  const client = useQueryClient();
  const locked = useRef(false);
  const [note, setNote] = useState<CloseRequest>({
    stopped_where: "",
    next_step: "",
    blocker: "",
    answer: "",
  });
  const close = useMutation({
    mutationFn: () => closeSession(session.session_id, note),
    retry: false,
    onSuccess: (result) => {
      client.setQueryData<TodayOut>(["today"], (previous) =>
        previous
          ? {
              ...previous,
              last_note: result.note,
              closed_today:
                previous.closed_today +
                // A lost response may have closed the session while this snapshot still says open.
                (previous.current?.session_id === session.session_id &&
                previous.last_note?.session_id !== result.note.session_id &&
                result.note.plan_date === previous.local_date
                  ? 1
                  : 0),
            }
          : previous,
      );
      onClosed(result, note);
      void client.invalidateQueries({ queryKey: ["today"], refetchType: "none" });
      for (const key of ["reviews", "progress", "coverage"])
        void client.invalidateQueries({ queryKey: [key] });
    },
    onSettled: () => {
      locked.current = false;
      onPendingChange(false);
    },
  });
  return (
    <section
      className="learning-panel close-panel"
      id="session-close"
      aria-labelledby="close-title"
    >
      <h2 id="close-title">When you stop</h2>
      <p>Leave your next session an exact place to begin.</p>
      <form
        className="learning-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (locked.current || disabled || !note.stopped_where.trim() || !note.next_step.trim())
            return;
          locked.current = true;
          onPendingChange(true);
          close.mutate();
        }}
      >
        {closeFields.map(([field, label]) => (
          <label key={field} htmlFor={`close-${field}`}>
            {label}
            <textarea
              id={`close-${field}`}
              required={field === "stopped_where" || field === "next_step"}
              disabled={close.isPending}
              rows={field === "answer" ? 4 : 2}
              value={note[field] ?? ""}
              onChange={(event) =>
                setNote((previous) => ({ ...previous, [field]: event.target.value }))
              }
            />
          </label>
        ))}
        {close.isError && <Notice tone="error">{describeError(close.error)}</Notice>}
        <button
          className="button"
          type="submit"
          disabled={
            disabled || close.isPending || !note.stopped_where.trim() || !note.next_step.trim()
          }
        >
          {close.isPending ? "Saving…" : "Save note and close"}
        </button>
      </form>
    </section>
  );
}
