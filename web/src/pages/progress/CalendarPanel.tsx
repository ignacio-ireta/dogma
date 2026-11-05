// The course calendar on the Progress page (ADR-0019). Every move is previewed first (a dry run),
// then confirmed. Nothing here counts days or judges the pace.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useId, useRef, useState } from "react";
import {
  applyWarp,
  type CalendarMove,
  type CalendarWarpOut,
  previewWarp,
} from "../../api/calendar";
import { describeError } from "../../api/client";
import { Loading } from "../../components/Loading";
import { Notice } from "../../components/Notice";
import "./calendar.css";
import { changeLines, furthestLine, offsetLine, thisWeekLine } from "./calendarText";
import { refreshCalendarData, useCalendar } from "./useCalendar";

interface Pending {
  readonly request: CalendarMove;
  readonly preview: CalendarWarpOut;
}

export function CalendarPanel() {
  const query = useCalendar();
  const client = useQueryClient();
  const sectionRef = useRef<HTMLElement>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [moved, setMoved] = useState<string | null>(null);
  const [goTo, setGoTo] = useState<number | null>(null);
  const gotoId = useId();

  useEffect(() => {
    // A link to /progress#calendar lands here; the section mounts after the page, so scroll once.
    if (window.location.hash === "#calendar") sectionRef.current?.scrollIntoView();
  }, []);

  const preview = useMutation({
    mutationFn: previewWarp,
    onSuccess: (result, request) => {
      setMoved(null);
      setPending({ request, preview: result });
    },
  });
  const apply = useMutation({
    mutationFn: (request: CalendarMove) => applyWarp(request),
    onSuccess: (result) => {
      setPending(null);
      setMoved(`Calendar moved. This week is week ${result.after.current_week}.`);
      void refreshCalendarData(client);
    },
  });

  function openPreview(request: CalendarMove) {
    setPending(null);
    preview.mutate(request);
  }

  const data = query.data;
  const busy = preview.isPending || apply.isPending;
  const failure = query.error ?? preview.error ?? apply.error;
  const offset = data ? offsetLine(data.offset_weeks) : null;

  return (
    <section className="calendar" id="calendar" aria-labelledby="calendar-title" ref={sectionRef}>
      <h2 id="calendar-title">Calendar</h2>
      {query.isPending && <Loading what="your calendar" />}
      {data && (
        <>
          <p className="calendar__week">
            {thisWeekLine(data.current_week, data.weeks.length, data.end)}
          </p>
          <p>{furthestLine(data.anchor_week, data.anchor_problem_id)}</p>
          {offset && <p className="muted">{offset}</p>}
          <div className="calendar__actions">
            <button
              type="button"
              className="button"
              disabled={busy || data.current_week === data.anchor_week}
              onClick={() => openPreview({ kind: "sync" })}
            >
              Match my exercises (week {data.anchor_week})
            </button>
          </div>
          <div className="calendar__goto">
            <label htmlFor={gotoId}>Go to week</label>
            <select
              id={gotoId}
              value={goTo ?? data.current_week}
              disabled={busy}
              onChange={(event) => setGoTo(Number(event.target.value))}
            >
              {data.weeks.map((item) => (
                <option key={item.week} value={item.week}>
                  Week {item.week}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="button"
              disabled={busy || (goTo ?? data.current_week) === data.current_week}
              onClick={() => openPreview({ kind: "to_week", week: goTo ?? data.current_week })}
            >
              Preview the move
            </button>
          </div>
          {data.offset_weeks !== 0 && (
            <div className="calendar__actions">
              <button
                type="button"
                className="button"
                disabled={busy}
                onClick={() => openPreview({ kind: "original" })}
              >
                Back to the original dates
              </button>
            </div>
          )}
        </>
      )}
      {preview.isPending && <Loading what="the preview" />}
      {pending && (
        <section className="calendar__preview" aria-labelledby="calendar-preview-title">
          <h3 id="calendar-preview-title">Preview</h3>
          {pending.request.kind === "sync" && (
            <p>
              {furthestLine(pending.preview.anchor_week, pending.preview.anchor_problem_id)}{" "}
              {pending.preview.anchor_problem_id === null
                ? `So this week becomes week ${pending.preview.after.current_week}.`
                : null}
            </p>
          )}
          {pending.preview.changed ? (
            <>
              <dl className="calendar__changes">
                {changeLines(pending.preview.before, pending.preview.after).map((line) => (
                  <div key={line.label}>
                    <dt>{line.label}</dt>
                    <dd>
                      {line.before} → {line.after}
                    </dd>
                  </div>
                ))}
              </dl>
              <div className="calendar__actions">
                <button
                  type="button"
                  className="button button--primary"
                  disabled={busy}
                  onClick={() => apply.mutate(pending.request)}
                >
                  Move the calendar
                </button>
                <button
                  type="button"
                  className="button"
                  disabled={busy}
                  onClick={() => setPending(null)}
                >
                  Cancel
                </button>
              </div>
            </>
          ) : (
            <>
              <p>
                This week is already week {pending.preview.before.current_week}. Nothing changed.
              </p>
              <button type="button" className="button" onClick={() => setPending(null)}>
                Close
              </button>
            </>
          )}
        </section>
      )}
      {moved && <Notice tone="success">{moved}</Notice>}
      {failure && <Notice tone="error">{describeError(failure)}</Notice>}
    </section>
  );
}
