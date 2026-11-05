import { readFileSync } from "node:fs";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "./Markdown";

const vectorLesson = readFileSync("../content/cml/vector-operations/lesson.md", "utf8");

describe("lesson Markdown", () => {
  it("renders the vector lesson's operations as a table with formulas in their cells", () => {
    const { container } = render(<Markdown source={vectorLesson} />);

    expect(container.querySelectorAll("table")).toHaveLength(1);
    const table = screen.getByRole("table");
    expect(
      within(table)
        .getAllByRole("columnheader")
        .map((cell) => cell.textContent),
    ).toEqual(["Operation", "Formula", "Shape in → out", "Meaning"]);
    expect(within(table).getAllByRole("row")).toHaveLength(6);
    const projection = within(table).getByText("Projection").closest("tr");
    expect(projection?.querySelectorAll("td")).toHaveLength(4);
    expect(projection?.querySelector("math annotation")).toHaveTextContent(
      "\\operatorname{proj}_{\\mathbf{b}}(\\mathbf{a})",
    );
  });

  it("renders the vector lesson's standalone equations in display mode", () => {
    const { container } = render(<Markdown source={vectorLesson} />);

    expect(container.querySelectorAll(".katex-display math")).toHaveLength(2);
    expect(container.querySelector(".katex-error")).toBeNull();
  });

  it.each(["$$x^2$$", "$$\nx^2\n$$", "> $$x^2$$", "- $$x^2$$"])(
    "renders standalone double-dollar math as a display equation: %s",
    (source) => {
      const { container } = render(<Markdown source={source} />);

      expect(container.querySelector(".katex-display math annotation")).toHaveTextContent("x^2");
    },
  );

  it("keeps inline equations inline and dollar signs inside code literal", () => {
    const { container } = render(
      <Markdown
        source={
          "A vector $\\mathbf{a}$ and $$b$$.\n\n$x^2$\n\n`$$literal$$`\n\n```text\n$$not_math$$\n```"
        }
      />,
    );

    expect(container.querySelectorAll(".katex math")).toHaveLength(3);
    expect(container.querySelector(".katex-display")).toBeNull();
    expect(screen.getByText("$$literal$$").tagName).toBe("CODE");
    expect(screen.getByText("$$not_math$$").tagName).toBe("CODE");
  });

  it("preserves table alignment, escaped pipes, and code while rendering math", () => {
    const { container } = render(
      <Markdown
        source={
          "| Left | Center | Right |\n| :--- | :---: | ---: |\n| `a\\|b` | $x^2$ | **Answer** |"
        }
      />,
    );

    expect(container.querySelectorAll("table")).toHaveLength(1);
    const cells = within(screen.getByRole("table")).getAllByRole("cell");
    expect(cells).toHaveLength(3);
    expect(cells[0]).toHaveTextContent("a|b");
    expect(cells[0]).toHaveStyle({ textAlign: "left" });
    expect(cells[1]).toHaveStyle({ textAlign: "center" });
    expect(cells[1]?.querySelector("math")).not.toBeNull();
    expect(cells[2]).toHaveStyle({ textAlign: "right" });
  });
});
