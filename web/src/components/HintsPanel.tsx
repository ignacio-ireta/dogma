// The hint ladder (ARCHITECTURE §4.5). Levels 1-3 are one click each. The walkthrough (level 4)
// needs an explicit confirmation first, and the API still enforces its own lock.
import { useEffect, useState } from "react";
import type { HintState } from "../api/hint-stream";
import { useHintStream } from "../hooks/useHintStream";
import { ConfirmDialog } from "./ConfirmDialog";
import { Markdown } from "./Markdown";
import { Notice } from "./Notice";

export const HINT_LEVELS = [
  { level: 1, label: "Concept" },
  { level: 2, label: "Approach" },
  { level: 3, label: "Code shape" },
] as const;
export const WALKTHROUGH_LEVEL = 4;

export interface HintsPanelProps {
  readonly problemId: string;
  readonly hintsCount: number;
  /** Ask for the next hint (⌘H). Registered by the page, so the shortcut and the buttons agree. */
  readonly onRegister: (nextLevel: () => void) => void;
  /** A timed re-solve of this problem is open: a hint makes it count as practice, so ask first. */
  readonly timedWindowOpen?: boolean;
}

export function nextHintLevel(current: number | null): number | null {
  const level = current === null ? 1 : current + 1;
  return level <= HINT_LEVELS.length ? level : null;
}

interface PendingHint {
  readonly level: number;
  readonly walkthrough: boolean;
}

export function HintsPanel({
  problemId,
  hintsCount,
  onRegister,
  timedWindowOpen = false,
}: HintsPanelProps) {
  const { state, request } = useHintStream(problemId);
  const [confirmed, setConfirmed] = useState(false);
  const [pending, setPending] = useState<PendingHint | null>(null);

  const ask = (level: number, walkthrough: boolean) => {
    if (timedWindowOpen) {
      setPending({ level, walkthrough });
      return;
    }
    void request(level, walkthrough);
  };

  const askNext = () => {
    if (hintsCount === 0 || state.phase === "streaming") return;
    const level = nextHintLevel(state.level);
    if (level !== null) {
      ask(level, false);
    }
  };
  useEffect(() => {
    onRegister(askNext);
  });

  if (hintsCount === 0) {
    return <p className="muted">This problem has no hints yet.</p>;
  }

  const busy = state.phase === "streaming";
  return (
    <div className="hints">
      <fieldset className="hints__buttons" aria-label="Ask for a hint">
        {HINT_LEVELS.map((item) => (
          <button
            key={item.level}
            type="button"
            className="button"
            disabled={busy}
            onClick={() => ask(item.level, false)}
          >
            Hint {item.level}: {item.label}
          </button>
        ))}
      </fieldset>
      {pending !== null ? (
        <ConfirmDialog
          title="Show a hint during the re-solve?"
          body="Showing a hint makes this re-solve count as practice, not a timed pass."
          confirmLabel="Show the hint"
          cancelLabel="Not now"
          onCancel={() => setPending(null)}
          onConfirm={() => {
            setPending(null);
            void request(pending.level, pending.walkthrough);
          }}
        />
      ) : null}
      <div className="hints__walkthrough">
        <label className="checkbox">
          <input
            type="checkbox"
            checked={confirmed}
            onChange={(event) => setConfirmed(event.target.checked)}
          />
          I want the full walkthrough. It shows the approach step by step, and it is locked until
          two submits have failed.
        </label>
        <button
          type="button"
          className="button"
          disabled={busy || !confirmed}
          onClick={() => ask(WALKTHROUGH_LEVEL, true)}
        >
          Show walkthrough (level 4)
        </button>
      </div>
      <HintOutcome state={state} />
    </div>
  );
}

function HintOutcome({ state }: { readonly state: HintState }) {
  if (state.phase === "idle") {
    return null;
  }
  const heading = state.level === null ? "Hint" : `Hint ${state.level}`;
  return (
    <article
      className="hint"
      aria-labelledby="hint-heading"
      aria-busy={state.phase === "streaming"}
    >
      <h2 id="hint-heading" className="hint__heading">
        {heading}
      </h2>
      {state.reviewed !== null ? (
        <section aria-label="Authored hint">
          <h3 className="hint__label">{state.reviewed.label}</h3>
          <Markdown source={state.reviewed.text} />
        </section>
      ) : null}
      {state.hintsNote !== null ? <p className="muted">{state.hintsNote}</p> : null}
      {state.phase === "streaming" && state.streamed !== "" ? (
        <p className="hint__streaming">{state.streamed}</p>
      ) : null}
      {state.phase === "done" && state.nudge !== null ? <Markdown source={state.nudge} /> : null}
      {state.phase === "done" && state.nudgeNote !== null ? (
        <p className="muted">{state.nudgeNote}</p>
      ) : null}
      {state.phase === "unavailable" ? (
        <Notice tone="info">
          The local model is not running, so there is no personal nudge. The hint above still
          stands.
          {state.unavailableNote !== null ? ` (${state.unavailableNote})` : ""}
        </Notice>
      ) : null}
      {state.phase === "error" && state.error !== null ? (
        <Notice tone="error">{state.error.detail}</Notice>
      ) : null}
      {nextHintLevel(state.level) !== null ? (
        <p className="muted hint__next">Press ⌘H for the next hint.</p>
      ) : null}
    </article>
  );
}
