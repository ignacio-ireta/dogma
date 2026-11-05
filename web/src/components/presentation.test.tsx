import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { attemptResult } from "../test/helpers";
import { Markdown } from "./Markdown";
import { ResultsPanel } from "./ResultsPanel";

it("renders public detail and only the API's safe hidden summary", () => {
  const { container } = render(<ResultsPanel result={attemptResult} error={null} running={null} />);
  expect(screen.getByText("test_public")).toBeVisible();
  expect(screen.getByText("test_hidden")).toBeVisible();
  expect(screen.getByText(/Full public detail/)).toBeVisible();
  expect(screen.getByText("Assertion failed (details hidden)")).toBeVisible();
  expect(screen.getByText("Recorded in your history.")).toBeVisible();
  expect(container.querySelectorAll(".test-row")).toHaveLength(2);
  expect(container.textContent).not.toMatch(/falsifying example|test_hidden\.py|Traceback/);
});

it.each(["rejected", "timeout", "oom", "error"] as const)(
  "shows %s diagnostics and recording warning",
  (status) => {
    render(
      <ResultsPanel
        result={{
          ...attemptResult,
          recorded: false,
          record_warning: "Scratch history is unavailable",
          report: {
            ...attemptResult.report,
            status,
            findings: ["Remove the banned import"],
            headline: "Tests did not finish",
            message: "Runner stopped safely",
            tests: [],
          },
        }}
        error={null}
        running={null}
      />,
    );
    expect(screen.getByText("Remove the banned import")).toBeVisible();
    expect(screen.getByText("Runner stopped safely")).toBeVisible();
    expect(screen.getByText(/Scratch history is unavailable/)).toBeVisible();
  },
);

it("hides old results while another attempt is pending", () => {
  render(<ResultsPanel result={attemptResult} error={null} running="submit" />);
  expect(screen.getByRole("status")).toHaveTextContent("Submitting");
  expect(screen.queryByText("test_hidden")).not.toBeInTheDocument();
});

describe("markdown", () => {
  it("renders bundled math and removes raw HTML and dangerous link targets", () => {
    const { container } = render(
      <Markdown
        source={
          '# Formula\n\n$\\alpha^2$\n\n<script>alert("unsafe")</script>\n\n<img src=x onerror="alert(1)">\n\n[safe](https://example.test) [bad](javascript:alert(1))'
        }
      />,
    );
    expect(screen.getByRole("heading", { name: "Formula" })).toBeVisible();
    expect(container.querySelector(".katex math")).not.toBeNull();
    expect(container.querySelector("script, img, [onerror]")).toBeNull();
    expect(screen.getByRole("link", { name: "safe" })).toHaveAttribute(
      "href",
      "https://example.test",
    );
    expect(screen.getByText("bad")).not.toHaveAttribute(
      "href",
      expect.stringContaining("javascript:"),
    );
  });
});
