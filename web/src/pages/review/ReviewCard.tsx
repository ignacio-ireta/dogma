import { type RefObject, useEffect, useRef } from "react";
import type { ReviewCardOut } from "../../api/learning";
import { Markdown } from "../../components/Markdown";

const ratings = ["Again", "Hard", "Good", "Easy"];

/** One due card: the question, an explicit reveal, then the four ratings. */
export function ReviewCard({
  card,
  size,
  reviewed,
  revealed,
  fetching,
  pending,
  stale,
  headingRef,
  onReveal,
  onRate,
}: {
  readonly card: ReviewCardOut;
  readonly size: number;
  readonly reviewed: number;
  readonly revealed: boolean;
  readonly fetching: boolean;
  readonly pending: boolean;
  readonly stale: boolean;
  readonly headingRef: RefObject<HTMLHeadingElement | null>;
  readonly onReveal: () => void;
  readonly onRate: (rating: number) => void;
}) {
  const answerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (revealed) answerRef.current?.focus();
  }, [revealed]);
  return (
    <section className="learning-panel" aria-labelledby="card-title" aria-busy={pending}>
      <p className="muted">
        {size} cards in this batch · {reviewed} reviewed
      </p>
      <h2 id="card-title" ref={headingRef} tabIndex={-1}>
        {card.topic_id}
      </h2>
      <Markdown source={card.question} />
      {!revealed ? (
        <button
          className="button button--primary"
          type="button"
          disabled={fetching}
          onClick={onReveal}
        >
          Reveal answer
        </button>
      ) : (
        <>
          <section
            className="review-answer"
            aria-labelledby="review-answer-title"
            ref={answerRef}
            tabIndex={-1}
          >
            <h3 id="review-answer-title">Answer</h3>
            <Markdown source={card.answer} />
          </section>
          <p className="muted">Choose how well you recalled it. Keys 1–4 also work.</p>
          <fieldset className="learning-actions" aria-label="Rate your recall">
            {ratings.map((label, index) => (
              <button
                className="button"
                type="button"
                key={label}
                disabled={pending || stale || fetching}
                onClick={() => onRate(index + 1)}
              >
                {index + 1} {label}
              </button>
            ))}
          </fieldset>
          {pending && !stale && <p role="status">Saving rating…</p>}
        </>
      )}
    </section>
  );
}
