// What the last Run or Submit said. It shows only what the API sent: hidden-test text arrives as a
// single safe line, so nothing here can reveal a hidden input.
import { describeError } from "../api/client";
import type { AttemptCreated, RunMode } from "../api/types";
import { countsFor, recordedLine, verdictLine } from "../lib/results";
import { Notice } from "./Notice";
import { TestRow } from "./TestRow";
import { TimedVerdictNote } from "./TimedVerdictNote";

export interface ResultsPanelProps {
  readonly result: AttemptCreated | undefined;
  readonly error: unknown;
  readonly running: RunMode | null;
}

export function ResultsPanel({ result, error, running }: ResultsPanelProps) {
  return (
    <section className="results" aria-labelledby="results-heading">
      <h2 id="results-heading" className="results__heading">
        Results
      </h2>
      {running !== null ? (
        <p className="results__running" role="status">
          {running === "submit" ? "Submitting" : "Running"} your file… This can take up to 30
          seconds.
        </p>
      ) : null}
      {error !== undefined && error !== null && running === null ? (
        <Notice tone="error">{describeError(error)}</Notice>
      ) : null}
      {result !== undefined && running === null ? <Verdict result={result} /> : null}
    </section>
  );
}

function Verdict({ result }: { readonly result: AttemptCreated }) {
  const report = result.report;
  const publicCounts = countsFor(report.tests, "public");
  const hiddenCounts = countsFor(report.tests, "hidden");
  const tone =
    report.status === "passed" ? "success" : report.status === "failed" ? "warning" : "error";
  return (
    <div className="verdict">
      <p className={`verdict__line verdict__line--${tone}`}>
        <span aria-hidden="true">{tone === "success" ? "✓" : tone === "warning" ? "•" : "✗"}</span>{" "}
        {verdictLine(report.status)}
      </p>
      <p className="verdict__counts">
        public {publicCounts.passed}/{publicCounts.total}
        {result.mode === "submit" ? (
          <>
            {" "}
            · hidden {hiddenCounts.passed}/{hiddenCounts.total}
          </>
        ) : null}
      </p>
      {report.findings.length > 0 ? (
        <div>
          <p className="verdict__subhead">Fix these, then run again:</p>
          <ul className="findings">
            {report.findings.map((finding) => (
              <li key={finding}>{finding}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {report.headline !== null ? <p className="verdict__note">{report.headline}.</p> : null}
      {report.message !== "" && report.status !== "failed" ? (
        <pre className="verdict__detail">{report.message}</pre>
      ) : null}
      {report.tests.length > 0 ? (
        <ul className="test-list" aria-label="Tests">
          {report.tests.map((test) => (
            <TestRow key={`${test.file}:${test.name}`} test={test} />
          ))}
        </ul>
      ) : null}
      {report.stdout !== "" ? (
        <details className="stdout">
          <summary>Your print output</summary>
          <pre>{report.stdout}</pre>
        </details>
      ) : null}
      {result.timed_resolve !== null ? <TimedVerdictNote verdict={result.timed_resolve} /> : null}
      <p className="muted results__recorded">{recordedLine(result)}</p>
    </div>
  );
}
