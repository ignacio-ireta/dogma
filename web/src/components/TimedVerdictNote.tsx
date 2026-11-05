// What this submit did to the timed re-solve it decided: the verdict in words, then the next due.
import type { TimedVerdictOut } from "../api/types";
import { nextDueLine, verdictLine } from "../lib/resolves";
import "../styles/resolves.css";

export function TimedVerdictNote({ verdict }: { readonly verdict: TimedVerdictOut }) {
  return (
    <div className="resolve-verdict">
      <p className="resolve-verdict__line" role="status">
        {verdictLine(verdict)}
      </p>
      {verdict.next_due !== null ? <p className="muted">{nextDueLine(verdict.next_due)}</p> : null}
    </div>
  );
}
