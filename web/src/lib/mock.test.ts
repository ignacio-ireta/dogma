import { describe, expect, it } from "vitest";
import { elapsedLabel, percentText, unscoredCount } from "./mock";

describe("unscoredCount", () => {
  it("counts the items still without a whole score from 1 to 4", () => {
    const criteria = ["a.one", "a.two", "b.one"];
    expect(unscoredCount(criteria, { "a.one": 4 })).toBe(2);
    expect(unscoredCount(criteria, { "a.one": 4, "a.two": 1, "b.one": 9 })).toBe(1);
    expect(unscoredCount(criteria, { "a.one": 4, "a.two": 1, "b.one": 2 })).toBe(0);
  });
});

describe("elapsedLabel", () => {
  const start = "2026-10-26T09:00:00Z";
  it("says how long the mock has run, calmly", () => {
    const at = Date.parse(start) + 14 * 60_000 + 30_000;
    expect(elapsedLabel(start, at)).toBe("14 min so far");
  });

  it("reads Just started in the first minute, and never warns", () => {
    expect(elapsedLabel(start, Date.parse(start) + 20_000)).toBe("Just started");
    expect(elapsedLabel("not a date", Date.now())).toBe("Just started");
  });
});

describe("percentText", () => {
  it("rounds to a whole percentage", () => {
    expect(percentText(66.67)).toBe("67%");
    expect(percentText(100)).toBe("100%");
  });
});
