import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, it } from "vitest";
import { NAV_ITEMS } from "../components/NavRail";
import { hasSeenWelcome } from "../lib/welcome";
import { FIRST_PROBLEM_TOUR_PATH, PLACES } from "../lib/welcomeContent";
import { WelcomePage } from "./WelcomePage";

beforeEach(() => window.localStorage.clear());

function renderPage() {
  render(
    <MemoryRouter>
      <WelcomePage />
    </MemoryRouter>,
  );
}

it("marks the introduction as seen, so the root opens Today next time", () => {
  renderPage();
  expect(hasSeenWelcome()).toBe(true);
});

it("gives one h1 and a heading for every section, in reading order", () => {
  renderPage();
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  expect(screen.getAllByRole("heading", { level: 2 }).map((item) => item.textContent)).toEqual([
    "What dogma is",
    "What a session looks like",
    "Where things live",
    "Your first 20 minutes",
    "Words you'll see",
    "When you're ready",
  ]);
});

it("shows the five session blocks with their minutes", () => {
  renderPage();
  const session = screen.getByRole("region", { name: "What a session looks like" });
  const blocks = within(session).getAllByRole("listitem");
  expect(blocks).toHaveLength(5);
  expect(blocks[0]).toHaveTextContent("Retrieve");
  expect(blocks[0]).toHaveTextContent("5 min");
  expect(blocks[3]).toHaveTextContent("Practice");
  expect(blocks[3]).toHaveTextContent("20 min");
});

it("offers a way past the page at the top and at the end", () => {
  renderPage();
  expect(screen.getByRole("link", { name: "Skip to Today" })).toHaveAttribute("href", "/today");
  expect(screen.getByRole("link", { name: "Go to Today" })).toHaveAttribute("href", "/today");
});

it("links the tutorial to the first problem with the guide on", () => {
  renderPage();
  expect(screen.getByRole("link", { name: "Try the first problem" })).toHaveAttribute(
    "href",
    FIRST_PROBLEM_TOUR_PATH,
  );
});

it("names every place the way the nav does, and links to it", () => {
  renderPage();
  const nav = new Map<string, string>(NAV_ITEMS.map((item) => [item.to, item.label]));
  for (const place of PLACES) {
    expect(nav.get(place.to)).toBe(place.label);
    expect(screen.getByRole("link", { name: new RegExp(`^${place.label}`) })).toHaveAttribute(
      "href",
      place.to,
    );
  }
});

it("covers everything in the nav except itself", () => {
  const covered = new Set(PLACES.map((place) => place.to));
  const missing = NAV_ITEMS.filter((item) => item.to !== "/welcome" && !covered.has(item.to));
  expect(missing).toEqual([]);
});
