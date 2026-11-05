// What a saved answer did. A promotion is named plainly; otherwise the server's reason is shown.
import type { AnswerOut } from "../../api/types";
import { Notice } from "../Notice";

const MASTERY_LABEL: Record<string, string> = {
  unseen: "not started",
  introduced: "introduced",
  solved: "solved",
  consolidating: "consolidating",
  retained: "retained",
  "interview-ready": "interview-ready",
};

export function AnswerResult({ result }: { readonly result: AnswerOut }) {
  const before = MASTERY_LABEL[result.before] ?? result.before;
  const after = MASTERY_LABEL[result.after] ?? result.after;
  return (
    <section className="answer-result" aria-labelledby="answer-result-title">
      <h3 id="answer-result-title">Answer saved</h3>
      {result.promoted ? (
        <Notice tone="success">This topic is now interview-ready.</Notice>
      ) : (
        <Notice tone="info">Mastery is still {after}.</Notice>
      )}
      <p className="muted">
        {before} → {after}
      </p>
      <p>{result.reason}</p>
    </section>
  );
}
