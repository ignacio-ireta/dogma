// Start over (ADR-0019): what is kept and cleared, then a dialog that needs the word `reset` typed.
// The backup is written before anything is cleared, so the result shows where it went.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { type ResetOut, startOver } from "../../api/calendar";
import { describeError } from "../../api/client";
import { Notice } from "../../components/Notice";
import "../../styles/resolves.css";
import "./calendar.css";
import { refreshCalendarData } from "./useCalendar";

const CONFIRM_WORD = "reset";

export function StartOverPanel() {
  const client = useQueryClient();
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [result, setResult] = useState<ResetOut | null>(null);
  const cancel = useRef<HTMLButtonElement>(null);
  const reset = useMutation({
    mutationFn: startOver,
    onSuccess: (done) => {
      setResult(done);
      setOpen(false);
      setTyped("");
      void refreshCalendarData(client);
    },
  });

  useEffect(() => {
    if (open) cancel.current?.focus();
  }, [open]);

  return (
    <section className="start-over" aria-labelledby="start-over-title">
      <h2 id="start-over-title">Start over</h2>
      <p>Starting over clears all of your progress:</p>
      <ul>
        <li>
          Cleared: attempts, topic mastery, study sessions and their notes, review cards and
          reviews, timed re-solves, mock and interview answers.
        </li>
        <li>Kept: the course content, the Library and your run folders.</li>
        <li>Workspace files are moved to var/workspace-history/, not deleted.</li>
        <li>This week becomes week 1, and a backup is written first.</li>
      </ul>
      {result && (
        <Notice tone="success">
          Your progress is cleared, and this week is week 1.
          <br />
          Backup: <code>{result.backup_path}</code>
          {result.snapshot_path && (
            <>
              <br />
              Database snapshot: <code>{result.snapshot_path}</code>
            </>
          )}
          <br />
          To restore the old progress, stop dogma serve and copy the snapshot over var/dogma.db.
        </Notice>
      )}
      {reset.isError && <Notice tone="error">{describeError(reset.error)}</Notice>}
      {open ? (
        <div className="confirm" role="alertdialog" aria-labelledby="start-over-dialog-title">
          <h3 className="confirm__title" id="start-over-dialog-title">
            Start over?
          </h3>
          <p className="confirm__body">
            Your progress is cleared after a backup is written. The course content is kept.
          </p>
          <div className="start-over__field">
            <label htmlFor="start-over-word">Type {CONFIRM_WORD} to start over</label>
            <input
              id="start-over-word"
              type="text"
              autoComplete="off"
              value={typed}
              disabled={reset.isPending}
              onChange={(event) => setTyped(event.target.value)}
            />
          </div>
          <div className="confirm__actions">
            <button
              type="button"
              className="button"
              ref={cancel}
              disabled={reset.isPending}
              onClick={() => {
                setOpen(false);
                setTyped("");
              }}
            >
              Keep my progress
            </button>
            <button
              type="button"
              className="button button--primary"
              disabled={typed !== CONFIRM_WORD || reset.isPending}
              onClick={() => reset.mutate()}
            >
              Clear progress and start over
            </button>
          </div>
        </div>
      ) : (
        <div className="calendar__actions">
          <button type="button" className="button" onClick={() => setOpen(true)}>
            Start over
          </button>
        </div>
      )}
    </section>
  );
}
