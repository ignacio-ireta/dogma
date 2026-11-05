import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { markWelcomeSeen } from "../lib/welcome";
import { RootRedirect } from "./RootRedirect";

beforeEach(() => window.localStorage.clear());
afterEach(() => vi.restoreAllMocks());

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route index element={<RootRedirect />} />
        <Route path="welcome" element={<p>welcome page</p>} />
        <Route path="today" element={<p>today page</p>} />
        <Route path="review" element={<p>review page</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

it("opens the introduction on a first visit to the root", () => {
  renderAt("/");
  expect(screen.getByText("welcome page")).toBeVisible();
});

it("opens Today once the introduction has been seen", () => {
  markWelcomeSeen();
  renderAt("/");
  expect(screen.getByText("today page")).toBeVisible();
});

it("falls back to Today when browser storage is blocked", () => {
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
    throw new DOMException("blocked", "SecurityError");
  });
  renderAt("/");
  expect(screen.getByText("today page")).toBeVisible();
});

it("never redirects a deep link on a first visit", () => {
  renderAt("/review");
  expect(screen.getByText("review page")).toBeVisible();
});
