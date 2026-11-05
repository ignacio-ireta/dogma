// The debrief of a scored mock: part totals, the band, the strongest part, the area to improve and
// the case's debrief text. Shown as reading, with no trend and no streak.
import type { MockDebriefOut, PartScoreOut } from "../../api/types";
import { Markdown } from "../../components/Markdown";
import { percentText } from "../../lib/mock";

export function DebriefPanel({ debrief }: { readonly debrief: MockDebriefOut }) {
  const report = debrief.report;
  return (
    <section className="mock-panel debrief" aria-labelledby="debrief-title">
      <h2 id="debrief-title">Debrief</h2>
      <p className="debrief__total">
        <strong>
          {report.total} of {report.maximum}
        </strong>{" "}
        · {report.band.label}
      </p>
      <p>{report.band.text}</p>
      <table className="learning-table">
        <caption className="muted">Part totals</caption>
        <thead>
          <tr>
            <th scope="col">Part</th>
            <th scope="col">Score</th>
            <th scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {report.parts.map((part) => (
            <tr key={part.part_id}>
              <th scope="row">{part.title}</th>
              <td>
                {part.score} of {part.maximum}
              </td>
              <td>{percentText(part.percent)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <dl className="debrief__notes">
        <dt>Strongest part</dt>
        <dd>{describePart(report.strongest)}</dd>
        <dt>Area to improve</dt>
        <dd>{describePart(report.weakest)}</dd>
      </dl>
      {debrief.case_debrief ? (
        <section aria-label="Case debrief">
          <h3>The case, a strong answer</h3>
          <Markdown source={debrief.case_debrief} />
        </section>
      ) : (
        <p className="muted">This mock has no case debrief text.</p>
      )}
    </section>
  );
}

function describePart(part: PartScoreOut): string {
  return `${part.title}, ${part.score} of ${part.maximum} (${percentText(part.percent)})`;
}
