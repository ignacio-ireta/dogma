import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { describeError } from "../api/client";
import { type CloseOut, type CloseRequest, getToday } from "../api/learning";
import { Icon } from "../components/Icon";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { PageHeading } from "../components/PageHeading";
import "../styles/learning.css";
import { CalendarOffer } from "./today/CalendarOffer";
import { CloseForm, closeFields } from "./today/CloseForm";
import { SessionPanel } from "./today/SessionPanel";
import { kindLabel } from "./today/sessionText";

export function TodayPage() {
  const [closed, setClosed] = useState<(CloseOut & { draft: CloseRequest }) | null>(null);
  const [closePending, setClosePending] = useState(false);
  const query = useQuery({
    queryKey: ["today"],
    queryFn: () => getToday(),
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  const data = query.data;
  const current = data?.current;
  const busy = query.isFetching || closePending;
  return (
    <div className="page learning-page today-page">
      <PageHeading
        title="Today"
        eyebrow="Big ideas. Your own pace."
        description="Follow your curiosity. Pick up an idea when you’re ready."
        illustrated
        topicId={current?.topic_id}
      />
      {query.isPending && <Loading what="your session" />}
      {query.isError && (
        <>
          <Notice tone="error">{describeError(query.error)}</Notice>
          <button className="button" type="button" onClick={() => void query.refetch()}>
            Try again
          </button>
        </>
      )}
      {data && (
        <>
          <CalendarOffer />
          <section className="learning-panel transition-panel" aria-labelledby="last-note-title">
            <span className="panel-icon">
              <Icon name="review" />
            </span>
            <h2 id="last-note-title">Last transition note</h2>
            {data.last_note ? (
              <>
                <p className="muted">
                  {data.last_note.plan_date} · {kindLabel[data.last_note.kind]}
                </p>
                <dl className="learning-note">
                  <dt>Stopped at</dt>
                  <dd>{data.last_note.stopped_where}</dd>
                  <dt>Next step</dt>
                  <dd>{data.last_note.next_step}</dd>
                  <dt>Blocker</dt>
                  <dd>{data.last_note.blocker || "None"}</dd>
                </dl>
              </>
            ) : (
              <p>No earlier session yet. Start with the first action below.</p>
            )}
          </section>
          {data.stale_open
            .filter((day) => day !== current?.plan_date)
            .map((day) => (
              <Notice key={day} tone="warning">
                The session from {day} was never closed, so it has no transition note.
              </Notice>
            ))}
          {closed ? (
            <ClosedPanel
              closed={closed}
              sessionsPerDay={data.sessions_per_day}
              closedToday={data.closed_today}
              busy={busy}
              onPlanNext={async () => {
                const result = await query.refetch();
                if (result.isSuccess) setClosed(null);
              }}
            />
          ) : current ? (
            <>
              <SessionPanel view={data} session={current} disabled={busy} />
              <CloseForm
                key={current.session_id}
                session={current}
                disabled={busy}
                onClosed={(result, draft) => setClosed({ ...result, draft })}
                onPendingChange={setClosePending}
              />
            </>
          ) : (
            <Notice tone="success">All sessions for today are closed. Come back tomorrow.</Notice>
          )}
        </>
      )}
    </div>
  );
}

function ClosedPanel({
  closed,
  sessionsPerDay,
  closedToday,
  busy,
  onPlanNext,
}: {
  readonly closed: CloseOut & { draft: CloseRequest };
  readonly sessionsPerDay: number;
  readonly closedToday: number;
  readonly busy: boolean;
  readonly onPlanNext: () => Promise<void>;
}) {
  return (
    <section className="learning-panel">
      {closed.already_closed ? (
        <>
          <Notice tone="warning">
            This session was already closed. The server kept the original note shown above. Review
            or copy your retry draft below before continuing.
          </Notice>
          <fieldset className="learning-form learning-actions">
            <legend>Your retry draft</legend>
            {closeFields.map(([field, label]) => (
              <label key={field}>
                {label}
                <textarea readOnly rows={2} value={closed.draft[field] ?? ""} />
              </label>
            ))}
          </fieldset>
        </>
      ) : (
        <Notice tone="success">Session closed. Your transition note is saved.</Notice>
      )}
      <p>Next step: {closed.note.next_step}</p>
      {closed.new_cards > 0 && <p>{closed.new_cards} new review cards enrolled.</p>}
      {closedToday < sessionsPerDay ? (
        <button className="button" type="button" disabled={busy} onClick={() => void onPlanNext()}>
          Plan next session
        </button>
      ) : (
        <p>All sessions for today are closed. Come back tomorrow.</p>
      )}
    </section>
  );
}
