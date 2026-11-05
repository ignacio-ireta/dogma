// Ask the interviewer something about the case. A reply is released only when the question matches
// the case's reveals; otherwise the reply says so, which is the rule, not a failure.
import { useState } from "react";
import { describeError } from "../../api/client";
import { Notice } from "../../components/Notice";
import type { useMockActions } from "../../hooks/useMock";

export function QuestionBox({
  reveal,
}: {
  readonly reveal: ReturnType<typeof useMockActions>["reveal"];
}) {
  const [question, setQuestion] = useState("");
  const ask = () => {
    const trimmed = question.trim();
    if (trimmed) reveal.mutate(trimmed);
  };
  return (
    <section className="mock-panel" aria-labelledby="mock-question-title">
      <h2 id="mock-question-title">Ask the interviewer</h2>
      <p className="muted">
        Put your question in your own words. Facts come only for a matching question.
      </p>
      <label className="mock-question__field">
        Your question
        <input
          type="text"
          value={question}
          maxLength={300}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") ask();
          }}
        />
      </label>
      <button
        className="button"
        type="button"
        disabled={reveal.isPending || !question.trim()}
        onClick={ask}
      >
        Ask
      </button>
      {reveal.isError && <Notice tone="error">{describeError(reveal.error)}</Notice>}
      {reveal.data && (
        <div className="mock-question__reply" role="status">
          <p>{reveal.data.text}</p>
          {reveal.data.exhibit && <p className="muted">Exhibit: {reveal.data.exhibit}</p>}
        </div>
      )}
    </section>
  );
}
