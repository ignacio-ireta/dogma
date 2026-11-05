// The planned minutes for each rubric part, shown as a plain list. A plan, not a deadline.
import type { MockPlanOut } from "../../api/types";

export function PartTimings({ parts }: { readonly parts: MockPlanOut["parts"] }) {
  return (
    <section className="mock-panel" aria-labelledby="mock-timings-title">
      <h2 id="mock-timings-title">Planned timings</h2>
      <ol className="mock-timings">
        {parts.map((part) => (
          <li key={part.title}>
            <span>{part.title}</span> <span className="muted">{part.minutes} min</span>
          </li>
        ))}
      </ol>
    </section>
  );
}
