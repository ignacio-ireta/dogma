// Readiness on core problems (ARCHITECTURE §4.4): the next timed re-solves and the last timed
// results. Counts of streaks, debt or overdue items are never shown (ADR-0013).
import type { ReadinessOut } from "../../api/types";
import { dueLine, timedResultLine } from "../../lib/resolves";
import "../../styles/resolves.css";

export function ReadinessPanel({ readiness }: { readonly readiness: ReadinessOut }) {
  return (
    <section className="readiness" aria-labelledby="readiness-title">
      <h2 id="readiness-title">Readiness</h2>
      <h3>Next timed re-solves</h3>
      {readiness.next_due.length === 0 ? (
        <p className="muted">
          No timed re-solve is scheduled yet. One is scheduled after you pass a core problem.
        </p>
      ) : (
        <ul className="readiness__list" aria-label="Next timed re-solves">
          {readiness.next_due.map((item) => (
            <li key={item.problem_id}>{dueLine(item)}</li>
          ))}
        </ul>
      )}
      <h3>Last timed results</h3>
      {readiness.recent_results.length === 0 ? (
        <p className="muted">No timed results yet.</p>
      ) : (
        <ul className="readiness__list" aria-label="Last timed results">
          {readiness.recent_results.map((item) => (
            <li key={`${item.problem_id}-${item.closed_at}`}>
              {item.problem_id} · {timedResultLine(item)}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
