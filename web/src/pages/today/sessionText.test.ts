import { describe, expect, it } from "vitest";
import type { SessionOut } from "../../api/learning";
import { browserInstruction } from "./sessionText";

// Only the fields browserInstruction reads.
const session = {
  topic_id: "cml/vector-operations",
  must_do: null,
  first_action: "",
  blocks: [],
} as unknown as SessionOut;

const problem = "cml/vector-operations/p01-vector-projection";

describe("browserInstruction for a timed re-solve", () => {
  it.each([
    [
      `Start the timed re-solve: dogma resolve ${problem} (budget 20 min, no hints).`,
      "Start the timed re-solve on the Problem page (budget 20 min, no hints).",
    ],
    [
      `Finish the open timed re-solve: dogma submit ${problem}, or dogma resolve ${problem} --abandon.`,
      "Finish the open timed re-solve: Submit, or abandon it on the Problem page.",
    ],
    [
      `Timed re-solves, no hints: Vector projection (${problem}), budget 20 min. Start with dogma resolve ${problem}, then dogma submit ${problem} when you have a solution.`,
      `Timed re-solves, no hints: Vector projection (${problem}), budget 20 min. Start on the Problem page, then Submit when you have a solution.`,
    ],
  ])("says the planned step in browser words: %#", (text, expected) => {
    expect(browserInstruction(text, session)).toBe(expected);
  });
});
