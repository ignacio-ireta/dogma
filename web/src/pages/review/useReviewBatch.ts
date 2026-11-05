import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ApiProblem } from "../../api/client";
import {
  type DueReviewsOut,
  getDueReviews,
  type ReviewCardOut,
  rateReview,
} from "../../api/learning";

export const queueKey = ["reviews", "due"];

/** The card changed, left the queue or was rated elsewhere: refresh the queue instead of writing again. */
function isStaleRating(error: unknown): boolean {
  return (
    error instanceof ApiProblem &&
    (error.problem.status === 409 || error.problem.code === "REVIEW_NOT_FOUND")
  );
}

/** Server state and interaction state for one bounded review batch. */
export function useReviewBatch() {
  const client = useQueryClient();
  const locked = useRef(false);
  const batchStart = useRef(0);
  const [revealedCard, setRevealedCard] = useState<ReviewCardOut | null>(null);
  const [staleCard, setStaleCard] = useState<ReviewCardOut | null>(null);
  const [reviewed, setReviewed] = useState(0);
  const [batchSize, setBatchSize] = useState(0);
  // Bumped whenever focus should move to the heading of whatever the learner now sees.
  const [focusRequest, setFocusRequest] = useState(0);
  const query = useQuery({
    queryKey: queueKey,
    queryFn: getDueReviews,
    retry: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  });
  const card = staleCard ?? query.data?.cards[0];
  const revealed =
    !!card &&
    revealedCard?.card_id === card.card_id &&
    revealedCard.due === card.due &&
    revealedCard.content_hash === card.content_hash;
  const rating = useMutation({
    mutationFn: ({ snapshot, value }: { snapshot: ReviewCardOut; value: number }) =>
      rateReview(snapshot.card_id, {
        rating: value,
        expected_due: snapshot.due,
        expected_content_hash: snapshot.content_hash,
      }),
    retry: false,
    onSuccess: async (_result, { snapshot }) => {
      // An older GET must not restore this card after its accepted rating.
      await client.cancelQueries({ queryKey: queueKey });
      client.setQueryData<DueReviewsOut>(queueKey, (previous) =>
        previous
          ? {
              ...previous,
              cards: previous.cards.filter((entry) => entry.card_id !== snapshot.card_id),
            }
          : previous,
      );
      setRevealedCard(null);
      setReviewed((count) => count + 1);
      setFocusRequest((count) => count + 1);
      for (const key of ["progress", "coverage"])
        void client.invalidateQueries({ queryKey: [key] });
    },
    onError: async (error, { snapshot }) => {
      if (isStaleRating(error)) {
        setStaleCard(snapshot);
        await query.refetch();
      }
    },
    onSettled: () => {
      locked.current = false;
    },
  });
  // A fresh batch starts when the queue loads after the previous one ran out.
  useEffect(() => {
    if (!query.data) return;
    const count = query.data.cards.length;
    if (batchStart.current === 0 && count > 0) {
      setBatchSize(count);
      setReviewed(0);
    }
    batchStart.current = count;
  }, [query.data]);
  const reveal = () => {
    if (card) setRevealedCard(card);
  };
  const submit = (value: number) => {
    if (!card || !revealed || staleCard || locked.current || rating.isPending || query.isFetching)
      return;
    locked.current = true;
    rating.mutate({ snapshot: card, value });
  };
  const refreshQueue = () => {
    setStaleCard(null);
    setRevealedCard(null);
    rating.reset();
    setFocusRequest((count) => count + 1);
  };
  return {
    query,
    rating,
    card,
    revealed,
    staleCard,
    reviewed,
    batchSize,
    focusRequest,
    reveal,
    submit,
    refreshQueue,
  };
}
