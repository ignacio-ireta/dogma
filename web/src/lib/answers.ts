// Pure rules for the Answer panel: when a self-score is complete, and what the request says.
import type { AnswerRequest } from "../api/types";

export type AnswerMode = "write" | "spoken";
export type Scores = Readonly<Record<string, number>>;

/** True when every criterion has a whole score from 1 to 4. Extra keys are never sent. */
export function scoresComplete(criteria: readonly string[], scores: Scores): boolean {
  return criteria.every((key) => {
    const value = scores[key];
    return value !== undefined && Number.isInteger(value) && value >= 1 && value <= 4;
  });
}

/** The request for one answer. A spoken answer sends no text, as `dogma answer --spoken` does. */
export function answerRequest(input: {
  readonly question: number;
  readonly mode: AnswerMode;
  readonly text: string;
  readonly scores: Scores;
}): AnswerRequest {
  const scores = { ...input.scores };
  if (input.mode === "spoken") {
    return { question: input.question, spoken: true, scores };
  }
  return { question: input.question, spoken: false, text: input.text, scores };
}

/** A written answer needs some words; a spoken one needs none. */
export function canSubmit(input: {
  readonly mode: AnswerMode;
  readonly text: string;
  readonly criteria: readonly string[];
  readonly scores: Scores;
}): boolean {
  if (!scoresComplete(input.criteria, input.scores)) return false;
  return input.mode === "spoken" || input.text.trim().length > 0;
}
