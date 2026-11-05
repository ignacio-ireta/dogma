import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router";
import { expect, it } from "vitest";
import { TourStrip } from "./TourStrip";

function Where() {
  const location = useLocation();
  return <output>{`${location.pathname}${location.search}`}</output>;
}

function renderAt(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <TourStrip />
      <Where />
    </MemoryRouter>,
  );
}

it("stays hidden on a normal visit to a problem", () => {
  renderAt("/problem/cml/x/p01");
  expect(screen.queryByRole("heading", { name: "Your first problem" })).toBeNull();
});

it("shows the steps when the tutorial link opens the problem", () => {
  renderAt("/problem/cml/x/p01?tour=1");
  expect(screen.getByRole("heading", { name: "Your first problem" })).toBeVisible();
  expect(screen.getAllByRole("listitem")).toHaveLength(4);
  expect(screen.getByText("Write your code and press Run")).toBeVisible();
});

it("hides and removes itself from the address, keeping other parameters", () => {
  renderAt("/problem/cml/x/p01?tour=1&other=2");
  fireEvent.click(screen.getByRole("button", { name: "Hide this guide" }));
  expect(screen.queryByRole("heading", { name: "Your first problem" })).toBeNull();
  expect(screen.getByRole("status")).toHaveTextContent("/problem/cml/x/p01?other=2");
});
