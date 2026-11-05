import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { expect, it } from "vitest";
import { NAV_ITEMS, NavRail } from "./NavRail";

it("draws a real icon for every nav item", () => {
  render(
    <MemoryRouter>
      <NavRail />
    </MemoryRouter>,
  );
  const nav = screen.getByRole("navigation", { name: "Main" });
  for (const item of NAV_ITEMS) {
    const link = within(nav).getByRole("link", { name: item.label });
    expect(link.querySelector("svg path")?.getAttribute("d")).toBeTruthy();
  }
});
