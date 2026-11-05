import { describe, expect, it } from "vitest";
import type { SessionOut } from "../../api/learning";
import { browserInstruction, firstActionTarget } from "./sessionText";

// Only the fields these helpers read.
const interview = {
  kind: "interview",
  topic_id: null,
  must_do: null,
  first_action: "Start the mock: dogma mock start (about 40 minutes, then the debrief).",
  blocks: [],
} as unknown as SessionOut;

describe("an Interview session", () => {
  it("links its first action to the Mock page", () => {
    expect(firstActionTarget(interview)).toBe("/mock");
  });

  it("says the mock steps in browser words, not CLI commands", () => {
    expect(browserInstruction(interview.first_action, interview)).toBe(
      "Start the mock on the Mock page (about 40 minutes, then the debrief).",
    );
    expect(
      browserInstruction(
        "dogma mock debrief shows your part totals, the band and the case's debrief. Then dogma close.",
        interview,
      ),
    ).toBe(
      "The Mock page's debrief shows your part totals, the band and the case's debrief. Then the close form below.",
    );
    expect(
      browserInstruction(
        "Timed: a case. Start with dogma mock start, then score with dogma mock score.",
        interview,
      ),
    ).toBe("Timed: a case. Start on the Mock page, then score there.");
  });
});
