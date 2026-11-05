// The Answer tab of a topic: pick a question, write or mark the answer as spoken, self-score it on
// the four dimensions with their anchors visible, and save. Saving shows what the answer did.
import { useId, useState } from "react";
import { describeError } from "../../api/client";
import type { AnswerOut } from "../../api/types";
import { useAnswerMutation, useInterview, useRubric } from "../../hooks/useInterview";
import { type AnswerMode, answerRequest, canSubmit, type Scores } from "../../lib/answers";
import { Loading } from "../Loading";
import { Notice } from "../Notice";
import { AnswerResult } from "./AnswerResult";
import { ScoreFields } from "./ScoreFields";

export function AnswerPanel({ topicId }: { readonly topicId: string }) {
  const interview = useInterview(topicId);
  const rubric = useRubric("topic-answer");
  const save = useAnswerMutation(topicId);
  const [question, setQuestion] = useState<number | null>(null);
  const [mode, setMode] = useState<AnswerMode>("write");
  const [text, setText] = useState("");
  const [scores, setScores] = useState<Scores>({});
  const [result, setResult] = useState<AnswerOut | null>(null);
  const ids = useId();

  if (interview.isError) return <Notice tone="error">{describeError(interview.error)}</Notice>;
  if (rubric.isError) return <Notice tone="error">{describeError(rubric.error)}</Notice>;
  const topicData = interview.data;
  const rubricData = rubric.data;
  if (topicData === undefined || rubricData === undefined) {
    return <Loading what="the interview questions" />;
  }
  const questions = topicData.questions;
  if (questions.length === 0) {
    return <p className="muted">This topic has no interview question yet.</p>;
  }
  const criteria = rubricData.criteria;
  const chosen = questions.find((item) => item.number === question) ?? questions[0];
  if (chosen === undefined) return null;
  const canSave = canSubmit({ mode, text, criteria, scores });

  const submit = () => {
    save.mutate(answerRequest({ question: chosen.number, mode, text, scores }), {
      onSuccess: (outcome) => setResult(outcome),
    });
  };

  return (
    <section className="answer-panel" aria-labelledby={`${ids}-title`}>
      <h3 id={`${ids}-title`}>Answer an interview question</h3>
      <p className="muted">
        Mastery: {interview.data.mastery.replaceAll("-", " ")}. Only a retained topic can become
        interview-ready.
      </p>
      <label className="answer-panel__field">
        Interview question
        <select
          value={chosen.number}
          disabled={save.isPending}
          onChange={(event) => {
            setQuestion(Number(event.target.value));
            setResult(null);
          }}
        >
          {questions.map((item) => (
            <option key={item.number} value={item.number}>
              Q{item.number}. {item.title}
            </option>
          ))}
        </select>
      </label>
      <fieldset className="answer-panel__mode">
        <legend>How did you answer?</legend>
        <label>
          <input
            type="radio"
            name={`${ids}-mode`}
            checked={mode === "write"}
            onChange={() => setMode("write")}
          />{" "}
          Write it
        </label>
        <label>
          <input
            type="radio"
            name={`${ids}-mode`}
            checked={mode === "spoken"}
            onChange={() => setMode("spoken")}
          />{" "}
          I answered it aloud (no text is kept)
        </label>
      </fieldset>
      {mode === "write" && (
        <label className="answer-panel__field">
          Your answer
          <textarea
            rows={6}
            value={text}
            disabled={save.isPending}
            onChange={(event) => setText(event.target.value)}
          />
        </label>
      )}
      <h4>Score yourself, 1 to 4 on each</h4>
      <ScoreFields
        dimensions={rubric.data.dimensions}
        scores={scores}
        disabled={save.isPending}
        onChange={(key, value) => setScores((current) => ({ ...current, [key]: value }))}
      />
      {save.isError && <Notice tone="error">{describeError(save.error)}</Notice>}
      <button
        className="button button--primary"
        type="button"
        disabled={!canSave || save.isPending}
        onClick={submit}
      >
        Save answer
      </button>
      {result && <AnswerResult result={result} />}
    </section>
  );
}
