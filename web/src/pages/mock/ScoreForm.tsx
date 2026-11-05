// The 30-item scoring form, grouped by part. The four scale anchors are shown once at the top, so
// every item is scored against them. The submit waits until every item has a score.
import { useState } from "react";
import { describeError } from "../../api/client";
import { Loading } from "../../components/Loading";
import { Notice } from "../../components/Notice";
import { useRubric } from "../../hooks/useInterview";
import type { useMockActions } from "../../hooks/useMock";
import type { Scores } from "../../lib/answers";
import { unscoredCount } from "../../lib/mock";

export function ScoreForm({
  actions,
  onScored,
}: {
  readonly actions: ReturnType<typeof useMockActions>;
  readonly onScored: (sessionId: string) => void;
}) {
  const rubric = useRubric("mock");
  const [scores, setScores] = useState<Scores>({});
  if (rubric.isPending) return <Loading what="the scoring rubric" />;
  if (rubric.isError) return <Notice tone="error">{describeError(rubric.error)}</Notice>;
  const left = unscoredCount(rubric.data.criteria, scores);
  const submit = () => {
    actions.score.mutate(scores as Record<string, number>, {
      onSuccess: (report) => onScored(report.session_id),
    });
  };
  return (
    <section className="mock-panel score-form" aria-labelledby="score-form-title">
      <h2 id="score-form-title">Score the mock</h2>
      <ol className="score-scale" aria-label="What each score means">
        {rubric.data.anchors.map((anchor) => (
          <li key={anchor.level}>
            <strong>{anchor.level}</strong> {anchor.text}
          </li>
        ))}
      </ol>
      {rubric.data.parts.map((part) => (
        <fieldset className="score-part" key={part.id}>
          <legend>
            {part.title}{" "}
            <span className="muted">
              · {part.minutes} min · max {part.maximum}
            </span>
          </legend>
          {part.items.map((item) => (
            <fieldset className="score-item" key={item.id}>
              <legend>{item.text}</legend>
              <div className="score-item__levels">
                {rubric.data.anchors.map((anchor) => (
                  <label key={anchor.level}>
                    <input
                      type="radio"
                      name={item.id}
                      value={anchor.level}
                      checked={scores[item.id] === anchor.level}
                      disabled={actions.score.isPending}
                      onChange={() =>
                        setScores((current) => ({ ...current, [item.id]: anchor.level }))
                      }
                    />{" "}
                    {anchor.level}
                  </label>
                ))}
              </div>
            </fieldset>
          ))}
        </fieldset>
      ))}
      {actions.score.isError && <Notice tone="error">{describeError(actions.score.error)}</Notice>}
      <p className="muted" role="status">
        {left === 0 ? "All 30 items scored." : `${left} items left to score.`}
      </p>
      <button
        className="button button--primary"
        type="button"
        disabled={left > 0 || actions.score.isPending}
        onClick={submit}
      >
        Score the mock
      </button>
    </section>
  );
}
