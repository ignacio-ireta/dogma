// The self-score for each dimension. Every score's anchor is printed beside its number, so the
// learner scores against the rubric's words. Each dimension is a radio group with its own name.
import { useId } from "react";
import type { RubricOut } from "../../api/types";
import type { Scores } from "../../lib/answers";

type Dimension = RubricOut["dimensions"][number];

export function ScoreFields({
  dimensions,
  scores,
  disabled,
  onChange,
}: {
  readonly dimensions: readonly Dimension[];
  readonly scores: Scores;
  readonly disabled: boolean;
  readonly onChange: (key: string, value: number) => void;
}) {
  const prefix = useId();
  return (
    <div className="score-fields">
      {dimensions.map((dimension) => (
        <fieldset className="score-field" key={dimension.id}>
          <legend>
            {dimension.title}
            <span className="score-field__question">{dimension.question}</span>
          </legend>
          {dimension.anchors.map((anchor) => {
            const id = `${prefix}-${dimension.id}-${anchor.level}`;
            return (
              <div className="score-option" key={anchor.level}>
                <input
                  id={id}
                  type="radio"
                  name={`${prefix}-${dimension.id}`}
                  value={anchor.level}
                  checked={scores[dimension.id] === anchor.level}
                  disabled={disabled}
                  onChange={() => onChange(dimension.id, anchor.level)}
                />
                <label htmlFor={id}>
                  <strong>{anchor.level}</strong> {anchor.text}
                </label>
              </div>
            );
          })}
        </fieldset>
      ))}
    </div>
  );
}
