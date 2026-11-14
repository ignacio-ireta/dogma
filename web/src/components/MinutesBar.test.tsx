import { render } from "@testing-library/react";
import { expect, it } from "vitest";
import { MinutesBar } from "./MinutesBar";

const blocks = [
  { name: "Retrieve", minutes: 5 },
  { name: "Predict", minutes: 3 },
  { name: "Learn", minutes: 12 },
];

it("draws one numbered segment per block, sized by its minutes, and states the real total", () => {
  const { container } = render(<MinutesBar blocks={blocks} />);
  const segments = [...container.querySelectorAll<HTMLElement>(".minutes-bar__segment")];
  expect(segments.map((item) => item.textContent)).toEqual(["1", "2", "3"]);
  expect(segments.map((item) => item.style.getPropertyValue("--minutes"))).toEqual([
    "5",
    "3",
    "12",
  ]);
  expect(container.querySelector("figcaption")).toHaveTextContent("20 min, drawn to scale");
});

it("stays out of the accessibility tree, because the rows beside it say the same thing", () => {
  const { container } = render(<MinutesBar blocks={blocks} />);
  expect(container.querySelector("figure")).toHaveAttribute("aria-hidden", "true");
  expect(container.querySelector("li, a, button")).toBeNull();
});

it("draws nothing when no block has time, and never sizes a segment below zero", () => {
  const empty = render(<MinutesBar blocks={[{ name: "Close", minutes: 0 }]} />);
  expect(empty.container).toBeEmptyDOMElement();
  const mixed = render(
    <MinutesBar
      blocks={[
        { name: "A", minutes: -4 },
        { name: "B", minutes: 10 },
      ]}
    />,
  );
  expect(mixed.container.querySelector("figcaption")).toHaveTextContent("10 min");
  expect(
    mixed.container
      .querySelector<HTMLElement>(".minutes-bar__segment")
      ?.style.getPropertyValue("--minutes"),
  ).toBe("0");
});
