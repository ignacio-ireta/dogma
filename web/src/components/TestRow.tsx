import type { LearnerTest } from "../api/types";
import { markFor } from "../lib/results";

/** One test: the mark, whether it is public or hidden, its name, and the message the API sent. */
export function TestRow({ test }: { readonly test: LearnerTest }) {
  const mark = markFor(test.outcome);
  return (
    <li className={`test-row test-row--${mark.tone}`}>
      <span className="test-row__mark" aria-label={mark.label} role="img">
        {mark.symbol}
      </span>
      <span className="test-row__head">
        <span className={`badge badge--${test.file}`}>{test.file}</span>
        <code className="test-row__name">{test.name}</code>
      </span>
      {test.message !== "" ? <pre className="test-row__message">{test.message}</pre> : null}
    </li>
  );
}
