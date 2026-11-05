import { describe, expect, it } from "vitest";
import { answerRequest, canSubmit, scoresComplete } from "./answers";

const criteria = ["accuracy", "structure", "trade-offs", "communication"];
const full = { accuracy: 3, structure: 4, "trade-offs": 2, communication: 3 };

describe("scoresComplete", () => {
  it("needs a whole score from 1 to 4 on every dimension", () => {
    expect(scoresComplete(criteria, full)).toBe(true);
    expect(scoresComplete(criteria, { structure: 4, "trade-offs": 2, communication: 3 })).toBe(
      false,
    );
    expect(scoresComplete(criteria, { ...full, accuracy: 0 })).toBe(false);
    expect(scoresComplete(criteria, { ...full, accuracy: 5 })).toBe(false);
    expect(scoresComplete(criteria, { ...full, accuracy: 2.5 })).toBe(false);
  });
});

describe("answerRequest", () => {
  it("sends the text for a written answer", () => {
    expect(answerRequest({ question: 2, mode: "write", text: "Alignment.", scores: full })).toEqual(
      { question: 2, spoken: false, text: "Alignment.", scores: full },
    );
  });

  it("sends no text for a spoken answer", () => {
    const body = answerRequest({ question: 1, mode: "spoken", text: "ignored", scores: full });
    expect(body).toEqual({ question: 1, spoken: true, scores: full });
    expect(body).not.toHaveProperty("text");
  });
});

describe("canSubmit", () => {
  it("waits for every score, and for words when the answer is written", () => {
    expect(canSubmit({ mode: "write", text: "  ", criteria, scores: full })).toBe(false);
    expect(canSubmit({ mode: "write", text: "Alignment.", criteria, scores: {} })).toBe(false);
    expect(canSubmit({ mode: "write", text: "Alignment.", criteria, scores: full })).toBe(true);
    expect(canSubmit({ mode: "spoken", text: "", criteria, scores: full })).toBe(true);
  });
});
