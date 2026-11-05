// Interview-ready topics per study week, and in total (I-18e). A count of what is ready, never a
// streak, a trend or a backlog (ADR-0013).
import type { InterviewReadyOut } from "../../api/types";

export function InterviewReadyPanel({ ready }: { readonly ready: InterviewReadyOut }) {
  return (
    <section className="interview-ready" aria-labelledby="interview-ready-title">
      <h2 id="interview-ready-title">Interview-ready</h2>
      <p>
        {ready.ready} of {ready.total} core topics are interview-ready.
      </p>
      <ul className="readiness__list" aria-label="Interview-ready topics per week">
        {ready.weeks.map((item) => (
          <li key={item.week}>
            Week {item.week} · interview-ready {item.ready} of {item.total} topics
          </li>
        ))}
      </ul>
    </section>
  );
}
