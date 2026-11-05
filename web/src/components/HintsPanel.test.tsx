import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { HintsPanel } from "./HintsPanel";

afterEach(() => vi.unstubAllGlobals());

it("requires explicit walkthrough confirmation and sends it to the API", async () => {
  const requests: unknown[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    requests.push(JSON.parse(String(init.body)));
    return new Response(
      'event: done\ndata: {"level":4,"reviewed":null,"nudge":null,"nudge_note":null}\n\n',
    );
  });
  render(<HintsPanel problemId="cml/topic/problem" hintsCount={4} onRegister={() => {}} />);
  const button = screen.getByRole("button", { name: /Show walkthrough/ });
  expect(button).toBeDisabled();
  fireEvent.click(screen.getByRole("checkbox"));
  fireEvent.click(button);
  await waitFor(() =>
    expect(requests).toEqual([
      { problem_id: "cml/topic/problem", level: 4, confirm_walkthrough: true },
    ]),
  );
  await screen.findByText("Hint 4");
});

it("does not request hints from the keyboard when the problem has no hints", () => {
  const fetch = vi.fn(async () => new Response());
  vi.stubGlobal("fetch", fetch);
  let next = () => {};
  render(
    <HintsPanel
      problemId="cml/topic/problem"
      hintsCount={0}
      onRegister={(callback) => {
        next = callback;
      }}
    />,
  );
  act(() => next());
  expect(fetch).not.toHaveBeenCalled();
});

it("keeps the next-hint keyboard action disabled while a hint is streaming", () => {
  const requests: unknown[] = [];
  vi.stubGlobal("fetch", (_url: string, init: RequestInit) => {
    requests.push(JSON.parse(String(init.body)));
    return new Promise(() => {});
  });
  let next = () => {};
  render(
    <HintsPanel
      problemId="cml/topic/problem"
      hintsCount={4}
      onRegister={(callback) => {
        next = callback;
      }}
    />,
  );
  act(() => next());
  expect(screen.getByRole("button", { name: "Hint 1: Concept" })).toBeDisabled();
  act(() => next());
  expect(requests).toEqual([
    { problem_id: "cml/topic/problem", level: 1, confirm_walkthrough: false },
  ]);
});

it("asks before a hint during a timed re-solve, and sends nothing until the learner agrees", async () => {
  const requests: unknown[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    requests.push(JSON.parse(String(init.body)));
    return new Response(
      'event: done\ndata: {"level":1,"reviewed":null,"nudge":null,"nudge_note":null}\n\n',
    );
  });
  render(
    <HintsPanel
      problemId="cml/topic/problem"
      hintsCount={4}
      timedWindowOpen
      onRegister={() => {}}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Hint 1: Concept" }));
  const dialog = screen.getByRole("alertdialog", { name: "Show a hint during the re-solve?" });
  expect(dialog).toHaveTextContent("makes this re-solve count as practice");
  expect(requests).toEqual([]);
  fireEvent.click(screen.getByRole("button", { name: "Not now" }));
  expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  expect(requests).toEqual([]);
  fireEvent.click(screen.getByRole("button", { name: "Hint 1: Concept" }));
  fireEvent.click(screen.getByRole("button", { name: "Show the hint" }));
  await waitFor(() =>
    expect(requests).toEqual([
      { problem_id: "cml/topic/problem", level: 1, confirm_walkthrough: false },
    ]),
  );
});

it("does not ask outside a timed re-solve", () => {
  const fetch = vi.fn(() => new Promise<Response>(() => {}));
  vi.stubGlobal("fetch", fetch);
  render(<HintsPanel problemId="cml/topic/problem" hintsCount={4} onRegister={() => {}} />);
  fireEvent.click(screen.getByRole("button", { name: "Hint 1: Concept" }));
  expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  expect(fetch).toHaveBeenCalledTimes(1);
});
