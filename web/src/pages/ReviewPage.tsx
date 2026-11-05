import { useEffect, useRef } from "react";
import { Link } from "react-router";
import { describeError } from "../api/client";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { PageHeading } from "../components/PageHeading";
import "../styles/learning.css";
import { ReviewCard } from "./review/ReviewCard";
import { useRatingKeys } from "./review/useRatingKeys";
import { useReviewBatch } from "./review/useReviewBatch";

export function ReviewPage() {
  const review = useReviewBatch();
  const { query, rating, card, revealed, staleCard, reviewed, batchSize, focusRequest } = review;
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (focusRequest > 0) headingRef.current?.focus();
  }, [focusRequest]);
  useRatingKeys(revealed && !staleCard, review.submit);
  return (
    <div className="page learning-page review-page">
      <PageHeading
        title="Review"
        eyebrow="Make it stay"
        description="A quiet moment to recall what you know. Think of the answer, then turn the card."
      />
      <p className="muted">A batch of up to 20 due cards. Recall the answer before revealing it.</p>
      {query.isPending && <Loading what="due cards" />}
      {query.isError && (
        <>
          <Notice tone="error">{describeError(query.error)}</Notice>
          <button
            className="button"
            type="button"
            disabled={query.isFetching}
            onClick={() => void query.refetch()}
          >
            Try again
          </button>
        </>
      )}
      {staleCard && (
        <section className="learning-panel" aria-label="Changed review card">
          <Notice tone="warning">
            This card’s schedule or content changed. Your previous answer remains below. Review the
            refreshed queue before rating again.
          </Notice>
          {query.isFetching ? (
            <Loading what="the updated queue" />
          ) : (
            !query.isError && (
              <button
                className="button"
                type="button"
                disabled={rating.isPending}
                onClick={review.refreshQueue}
              >
                Use updated queue
              </button>
            )
          )}
        </section>
      )}
      {rating.isError && !staleCard && <Notice tone="error">{describeError(rating.error)}</Notice>}
      {card ? (
        <ReviewCard
          card={card}
          size={batchSize}
          reviewed={reviewed}
          revealed={revealed}
          fetching={query.isFetching}
          pending={rating.isPending}
          stale={staleCard !== null}
          headingRef={headingRef}
          onReveal={review.reveal}
          onRate={review.submit}
        />
      ) : (
        query.isSuccess && (
          <section className="learning-panel">
            <h2 ref={headingRef} tabIndex={-1}>
              {reviewed ? "This review batch is complete." : "Nothing is due right now."}
            </h2>
            <p>
              {reviewed
                ? "You can stop here or check for another batch."
                : "Study a ready topic and close its session to bring its cards into review."}
            </p>
            <div className="learning-actions">
              <Link className="button" to="/today">
                Back to Today
              </Link>
              <button
                className="button"
                type="button"
                disabled={query.isFetching}
                onClick={() => void query.refetch()}
              >
                Check for due cards
              </button>
            </div>
          </section>
        )
      )}
    </div>
  );
}
