import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router";
import { describeError, encodeId } from "../../api/client";
import { type Energy, getToday, type SessionOut, type TodayOut } from "../../api/learning";
import { Icon } from "../../components/Icon";
import { MinutesBar } from "../../components/MinutesBar";
import { Notice } from "../../components/Notice";
import { actionProblem, browserInstruction, firstActionTarget, kindLabel } from "./sessionText";
import { TimedResolveAction } from "./TimedResolveAction";

/** The open session: its kind, first action, blocks and the energy controls that replan it. */
export function SessionPanel({
  view,
  session,
  disabled,
}: {
  readonly view: TodayOut;
  readonly session: SessionOut;
  readonly disabled: boolean;
}) {
  const client = useQueryClient();
  const energy = useMutation({
    mutationFn: (mode: Energy) => getToday(mode),
    retry: false,
    onSuccess: (next) => client.setQueryData(["today"], next),
  });
  const busy = disabled || energy.isPending;
  const familiar = session.blocks.map((block) => actionProblem(block.detail)).find(Boolean);
  return (
    <section className="learning-panel session-panel" aria-labelledby="session-title">
      <p className="muted">
        {view.local_date} · Session {session.slot} of {session.sessions_per_day}
      </p>
      <h2 id="session-title">
        {kindLabel[session.kind]} · {session.minutes} min
      </h2>
      {view.replanned ? (
        <p role="status">Replanned for {session.energy} energy.</p>
      ) : (
        view.resumed && <p>Resuming your open session.</p>
      )}
      {session.plan_date !== view.local_date && (
        <Notice tone="info">
          Originally planned for {session.plan_date}. Close this session before starting a new one.
        </Notice>
      )}
      <fieldset className="learning-actions" aria-label="Energy mode">
        {(["normal", "low"] as const).map((mode) => (
          <button
            className="button"
            type="button"
            key={mode}
            aria-pressed={session.energy === mode}
            disabled={busy || mode === session.energy}
            onClick={() => energy.mutate(mode)}
          >
            {mode === "normal" ? "Normal energy" : "Low energy"}
          </button>
        ))}
      </fieldset>
      {energy.isError && <Notice tone="error">{describeError(energy.error)}</Notice>}
      <p className="learning-first-action">
        <strong>First action:</strong>{" "}
        <span>{browserInstruction(session.first_action, session)}</span>
      </p>
      {session.timed_resolve ? (
        <TimedResolveAction timed={session.timed_resolve} disabled={busy} />
      ) : firstActionTarget(session) === "#session-close" ? (
        <a className="button button--primary learning-start" href="#session-close">
          Finish session
        </a>
      ) : (
        <Link className="button button--primary learning-start" to={firstActionTarget(session)}>
          {session.kind === "interview" ? "Start the mock" : "Start session"}
          <Icon name="arrow" />
        </Link>
      )}
      <MinutesBar blocks={session.blocks} />
      <div className="learning-table-wrap">
        <table className="learning-table">
          <caption className="muted">Your session blocks</caption>
          <thead>
            <tr>
              <th scope="col">Block</th>
              <th scope="col">Time</th>
              <th scope="col">What to do</th>
            </tr>
          </thead>
          <tbody>
            {session.blocks.map((block) => (
              <tr key={block.name}>
                <th scope="row">{block.name}</th>
                <td>{block.minutes} min</td>
                <td>{browserInstruction(block.detail, session)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {session.notes.map((text) => (
        <Notice key={text} tone="info">
          {browserInstruction(text, session)}
        </Notice>
      ))}
      {session.must_do && (
        <p className="learning-must-do">
          One must-do: <code>{session.must_do}</code>. Everything else is a bonus.
        </p>
      )}
      {session.must_do && firstActionTarget(session) === "/review" && (
        <p>
          <Link to={`/problem/${encodeId(session.must_do)}`}>Open core problem</Link>
        </p>
      )}
      {!session.must_do && familiar && firstActionTarget(session) === "/review" && (
        <p>
          <Link to={`/problem/${encodeId(familiar)}`}>Open familiar problem</Link>
        </p>
      )}
      {session.must_do && /Predict/.test(session.first_action) && (
        <p className="muted">Write your guess first, then use the problem’s Lesson tab.</p>
      )}
    </section>
  );
}
