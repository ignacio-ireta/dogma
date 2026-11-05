import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { hasSeenWelcome, markWelcomeSeen } from "./welcome";

beforeEach(() => window.localStorage.clear());
afterEach(() => vi.restoreAllMocks());

it("has not been seen until it is marked", () => {
  expect(hasSeenWelcome()).toBe(false);
  markWelcomeSeen();
  expect(hasSeenWelcome()).toBe(true);
});

it("treats unreadable storage as seen, so the learner goes to Today", () => {
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
    throw new DOMException("blocked", "SecurityError");
  });
  expect(hasSeenWelcome()).toBe(true);
});

it("does not throw when storage cannot be written", () => {
  vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
    throw new DOMException("full", "QuotaExceededError");
  });
  expect(() => markWelcomeSeen()).not.toThrow();
});
