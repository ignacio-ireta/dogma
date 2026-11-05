import { apiFetch, encodeId } from "./client";
import type { components } from "./schema";

export type TodayOut = components["schemas"]["TodayOut"];
export type SessionOut = components["schemas"]["SessionOut"];
export type CloseRequest = components["schemas"]["CloseRequest"];
export type CloseOut = components["schemas"]["CloseOut"];
export type ReviewCardOut = components["schemas"]["ReviewCardOut"];
export type DueReviewsOut = components["schemas"]["DueReviewsOut"];
export type RatingRequest = components["schemas"]["RatingRequest"];
export type ReviewOut = components["schemas"]["ReviewOut"];
export type Energy = components["schemas"]["Energy"];

// Omitting energy resumes the stored mode, including a low-energy session.
export function getToday(energy?: Energy): Promise<TodayOut> {
  return apiFetch<TodayOut>(`/today${energy ? `?energy=${energy}` : ""}`);
}

export function closeSession(sessionId: string, note: CloseRequest): Promise<CloseOut> {
  return apiFetch<CloseOut>(`/sessions/${encodeURIComponent(sessionId)}/close`, {
    method: "POST",
    body: JSON.stringify(note),
  });
}

export function getDueReviews(): Promise<DueReviewsOut> {
  return apiFetch<DueReviewsOut>("/reviews/due?limit=20");
}

export function rateReview(cardId: string, rating: RatingRequest): Promise<ReviewOut> {
  return apiFetch<ReviewOut>(`/reviews/${encodeId(cardId)}`, {
    method: "POST",
    body: JSON.stringify(rating),
  });
}
