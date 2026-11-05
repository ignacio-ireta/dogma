import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import type { components } from "../api/schema";
import { deferred, jsonResponse, queryWrapper } from "../test/helpers";
import { ReviewPage } from "./ReviewPage";
import { TodayPage } from "./TodayPage";

type Today = components["schemas"]["TodayOut"];
const note: components["schemas"]["NoteOut"] = {
  session_id: "old-session",
  plan_date: "2026-10-24",
  slot: 1,
  kind: "learn",
  topic_id: "cml/topic",
  stopped_where: "At the worked example",
  next_step: "Try the gradient",
  blocker: "Check the dimensions",
  closed_at: "2026-10-24T12:00:00Z",
};
const today: Today = {
  local_date: "2026-10-25",
  sessions_per_day: 3,
  closed_today: 0,
  last_note: note,
  resumed: true,
  replanned: false,
  stale_open: [],
  current: {
    session_id: "scratch-session",
    plan_date: "2026-10-25",
    slot: 1,
    sessions_per_day: 3,
    kind: "learn",
    energy: "normal",
    topic_id: "cml/topic",
    must_do: "cml/topic/p01",
    first_action: "Write your Predict guess (3 min), then open content/cml/topic/lesson.md.",
    minutes: 45,
    notes: [],
    timed_resolve: null,
    blocks: [
      { name: "Retrieve", minutes: 5, detail: "Retrieve past material" },
      { name: "Predict", minutes: 3, detail: "Make a prediction" },
      { name: "Learn", minutes: 12, detail: "Read the lesson" },
      { name: "Practice", minutes: 20, detail: "Solve the core problem" },
      { name: "Close", minutes: 5, detail: "Leave a note" },
    ],
  },
};
const card: components["schemas"]["ReviewCardOut"] = {
  card_id: "cml/topic#1",
  topic_id: "cml/topic",
  question: "Recall **the gradient**: [...]",
  answer: "The gradient is **a vector**.",
  due: "2026-10-25T00:00:00Z",
  content_hash: "old-hash",
};
function failure(status: number, detail: string) {
  return jsonResponse(
    {
      type: "about:blank",
      title: "Failed",
      status,
      detail,
      code: "API_FAILURE",
      instance: "",
      errors: null,
    },
    status,
  );
}
function page(element: React.ReactNode) {
  return render(<MemoryRouter>{element}</MemoryRouter>, { wrapper: queryWrapper() });
}
afterEach(() => vi.unstubAllGlobals());

it("offers today's first session after closing yesterday's final slot", async () => {
  const oldSession = {
    ...today.current,
    session_id: "scratch-session",
    plan_date: "2026-10-24",
    slot: 3,
  };
  const savedNote = { ...note, session_id: oldSession.session_id, slot: 3 };
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) =>
    jsonResponse(
      init.method === "POST"
        ? { note: savedNote, already_closed: false, introduced_topic: null, new_cards: 0 }
        : { ...today, current: oldSession, stale_open: ["2026-10-24"] },
    ),
  );
  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <TodayPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
  await screen.findByLabelText("Where I stopped");
  fireEvent.change(screen.getByLabelText("Where I stopped"), { target: { value: "end" } });
  fireEvent.change(screen.getByLabelText("Exact next step"), {
    target: { value: "today's start" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
  expect(await screen.findByRole("button", { name: "Plan next session" })).toBeVisible();
  expect(client.getQueryData<Today>(["today"])?.closed_today).toBe(0);
  expect(
    screen.queryByText("All sessions for today are closed. Come back tomorrow."),
  ).not.toBeInTheDocument();
});

it("does not restore a successfully rated card from an older in-flight queue response", async () => {
  const refresh = deferred<Response>();
  const write = deferred<Response>();
  let loads = 0;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") return write.promise;
    return ++loads === 1 ? jsonResponse({ enrolled: 0, cards: [card] }) : refresh.promise;
  });
  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ReviewPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  act(() => {
    void client.invalidateQueries({ queryKey: ["reviews"] });
  });
  await waitFor(() => expect(loads).toBe(2));
  await act(async () =>
    write.resolve(
      jsonResponse({
        card_id: card.card_id,
        topic_id: card.topic_id,
        due: "2026-10-26T00:00:00Z",
        rating: 3,
        mastery: "introduced",
      }),
    ),
  );
  expect(
    await screen.findByRole("heading", { name: "This review batch is complete." }),
  ).toBeVisible();
  await act(async () => refresh.resolve(jsonResponse({ enrolled: 0, cards: [card] })));
  await waitFor(() => expect(client.isFetching({ queryKey: ["reviews"] })).toBe(0));
  expect(screen.getByRole("heading", { name: "This review batch is complete." })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Reveal answer" })).not.toBeInTheDocument();
});

it("preserves an edited retry draft when the server already closed the session", async () => {
  const pending = deferred<Response>();
  let writes = 0;
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") {
      return ++writes === 1
        ? pending.promise
        : jsonResponse({
            note: { ...note, plan_date: "2026-10-25", session_id: "scratch-session" },
            already_closed: true,
            introduced_topic: null,
            new_cards: 0,
          });
    }
    return jsonResponse(today);
  });
  page(<TodayPage />);
  await screen.findByLabelText("Where I stopped");
  fireEvent.change(screen.getByLabelText("Where I stopped"), { target: { value: "original" } });
  fireEvent.change(screen.getByLabelText("Exact next step"), {
    target: { value: "original next" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
  await act(async () => pending.reject(new TypeError("Lost response after commit")));
  await screen.findByText(/local API is not reachable/);
  fireEvent.change(screen.getByLabelText("Where I stopped"), { target: { value: "edited stop" } });
  fireEvent.change(screen.getByLabelText("Exact next step"), { target: { value: "edited next" } });
  fireEvent.change(screen.getByLabelText("Blocker (optional)"), {
    target: { value: "edited blocker" },
  });
  fireEvent.change(screen.getByLabelText("60-second answer (optional)"), {
    target: { value: "edited answer" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
  expect(await screen.findByText(/already closed/)).toBeVisible();
  for (const draft of ["edited stop", "edited next", "edited blocker", "edited answer"]) {
    expect(screen.getByDisplayValue(draft)).toBeVisible();
  }
  expect(
    screen.queryByText("Session closed. Your transition note is saved."),
  ).not.toBeInTheDocument();
});

it.each([false, true])(
  "marks today's final slot complete after close (already_closed=%s)",
  async (alreadyClosed) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
    vi.stubGlobal("fetch", async (_url: string, init: RequestInit) =>
      jsonResponse(
        init.method === "POST"
          ? {
              note: { ...note, plan_date: "2026-10-25", slot: 3, session_id: "scratch-session" },
              already_closed: alreadyClosed,
              introduced_topic: null,
              new_cards: 0,
            }
          : { ...today, closed_today: 2, current: { ...today.current, slot: 3 } },
      ),
    );
    render(
      <MemoryRouter>
        <QueryClientProvider client={client}>
          <TodayPage />
        </QueryClientProvider>
      </MemoryRouter>,
    );
    await screen.findByLabelText("Where I stopped");
    fireEvent.change(screen.getByLabelText("Where I stopped"), { target: { value: "end" } });
    fireEvent.change(screen.getByLabelText("Exact next step"), { target: { value: "tomorrow" } });
    fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
    expect(
      await screen.findByText("All sessions for today are closed. Come back tomorrow."),
    ).toBeVisible();
    expect(client.getQueryData<Today>(["today"])?.closed_today).toBe(3);
    expect(screen.queryByRole("button", { name: "Plan next session" })).not.toBeInTheDocument();
  },
);

it("shows the last note before the exact session blocks and one browser first action", async () => {
  vi.stubGlobal("fetch", async () => jsonResponse(today));
  page(<TodayPage />);
  expect(screen.getByRole("status")).toHaveTextContent("Loading");
  const last = await screen.findByRole("heading", { name: "Last transition note" });
  const plan = screen.getByRole("heading", { name: "Learn · 45 min" });
  expect(last.compareDocumentPosition(plan) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(screen.getByText(note.next_step)).toBeVisible();
  expect(screen.getByText("Resuming your open session.")).toBeVisible();
  const rows = within(screen.getByRole("table")).getAllByRole("row").slice(1);
  expect(rows.map((row) => row.textContent)).toEqual(
    today.current?.blocks.map((b) => `${b.name}${b.minutes} min${b.detail}`),
  );
  expect(
    screen.getByText("Write your Predict guess (3 min), then open the Lesson tab."),
  ).toBeVisible();
  expect(screen.getAllByRole("link", { name: "Start session" })).toHaveLength(1);
  expect(screen.getByRole("link", { name: "Start session" })).toHaveAttribute(
    "href",
    "/problem/cml/topic/p01",
  );
});

it("resumes without an energy override and replans only on explicit energy controls", async () => {
  const fetcher = vi.fn(async (url: string) =>
    jsonResponse(
      url.includes("energy=low")
        ? {
            ...today,
            replanned: true,
            current: {
              ...today.current,
              energy: "low",
              kind: "minimum",
              minutes: 20,
              must_do: null,
              first_action: "Finish with dogma close, then rest. Nothing else is planned.",
              blocks: [
                { name: "Review", minutes: 10, detail: "Nothing due" },
                { name: "Re-solve", minutes: 10, detail: "Nothing familiar" },
              ],
            },
          }
        : today,
    ),
  );
  vi.stubGlobal("fetch", fetcher);
  page(<TodayPage />);
  await screen.findByText("Learn · 45 min");
  expect(fetcher.mock.calls[0]?.[0]).toBe("/api/v1/today");
  fireEvent.click(screen.getByRole("button", { name: "Low energy" }));
  await screen.findByText("Minimum viable · 20 min");
  expect(fetcher.mock.calls.at(-1)?.[0]).toBe("/api/v1/today?energy=low");
  expect(screen.getByRole("link", { name: "Finish session" })).toHaveAttribute(
    "href",
    "#session-close",
  );
  fireEvent.click(screen.getByRole("button", { name: "Normal energy" }));
  await screen.findByText("Learn · 45 min");
  expect(fetcher.mock.calls.at(-1)?.[0]).toBe("/api/v1/today?energy=normal");
});

it("keeps every close field after failure and closes without claiming a solve", async () => {
  const pending = deferred<Response>();
  const writes: RequestInit[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") {
      writes.push(init);
      return writes.length === 1
        ? pending.promise
        : jsonResponse({
            note,
            already_closed: false,
            introduced_topic: "cml/topic",
            new_cards: 2,
          });
    }
    return jsonResponse(today);
  });
  page(<TodayPage />);
  await screen.findByText("Learn · 45 min");
  for (const [label, value] of [
    ["Where I stopped", "at line 4"],
    ["Exact next step", "check shape"],
    ["Blocker (optional)", "dimension"],
    ["60-second answer (optional)", "my explanation"],
  ]) {
    fireEvent.change(screen.getByLabelText(label ?? ""), { target: { value } });
  }
  fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
  expect(await screen.findByRole("button", { name: "Saving…" })).toBeDisabled();
  await act(async () => pending.resolve(failure(500, "Scratch write failed")));
  expect(await screen.findByText("Scratch write failed")).toBeVisible();
  expect(screen.getByLabelText("Where I stopped")).toHaveValue("at line 4");
  expect(screen.getByLabelText("Exact next step")).toHaveValue("check shape");
  expect(screen.getByLabelText("Blocker (optional)")).toHaveValue("dimension");
  expect(screen.getByLabelText("60-second answer (optional)")).toHaveValue("my explanation");
  fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
  expect(await screen.findByText("Session closed. Your transition note is saved.")).toBeVisible();
  expect(JSON.parse(String(writes[1]?.body))).toEqual({
    stopped_where: "at line 4",
    next_step: "check shape",
    blocker: "dimension",
    answer: "my explanation",
  });
  expect(screen.queryByText(/topic solved/i)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Plan next session" })).toBeVisible();
});

it("shows completed days honestly without a start or close action", async () => {
  vi.stubGlobal("fetch", async () => jsonResponse({ ...today, current: null, closed_today: 3 }));
  page(<TodayPage />);
  expect(
    await screen.findByText("All sessions for today are closed. Come back tomorrow."),
  ).toBeVisible();
  expect(screen.queryByRole("link", { name: "Start session" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Save note and close" })).not.toBeInTheDocument();
});

it("shows first-session and load-error states with retry", async () => {
  let loads = 0;
  vi.stubGlobal("fetch", async () =>
    ++loads === 1 ? failure(500, "Plan unavailable") : jsonResponse({ ...today, last_note: null }),
  );
  page(<TodayPage />);
  expect(await screen.findByText("Plan unavailable")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(
    await screen.findByText("No earlier session yet. Start with the first action below."),
  ).toBeVisible();
});

it("routes a review first action into the browser", async () => {
  vi.stubGlobal("fetch", async () =>
    jsonResponse({
      ...today,
      current: { ...today.current, first_action: "Run dogma review --limit 5." },
    }),
  );
  page(<TodayPage />);
  expect(await screen.findByRole("link", { name: "Start session" })).toHaveAttribute(
    "href",
    "/review",
  );
});

it("reveals Markdown answers and binds one explicit rating to the displayed snapshot", async () => {
  const pending = deferred<Response>();
  const writes: { url: string; body: unknown }[] = [];
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    if (init.method === "POST") {
      writes.push({ url, body: JSON.parse(String(init.body)) });
      return pending.promise;
    }
    return jsonResponse({ enrolled: 1, cards: [card] });
  });
  page(<ReviewPage />);
  expect(await screen.findByText("the gradient")).toBeVisible();
  expect(screen.queryByText("a vector")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Good/ })).not.toBeInTheDocument();
  fireEvent.keyDown(window, { key: "3" });
  expect(writes).toHaveLength(0);
  fireEvent.click(screen.getByRole("button", { name: "Reveal answer" }));
  expect(screen.getByText("a vector").tagName).toBe("STRONG");
  fireEvent.keyDown(window, { key: "3", repeat: true });
  expect(writes).toHaveLength(0);
  fireEvent.keyDown(window, { key: "3" });
  fireEvent.keyDown(window, { key: "3" });
  await waitFor(() => expect(screen.getByRole("button", { name: "3 Good" })).toBeDisabled());
  await waitFor(() => expect(writes).toHaveLength(1));
  expect(writes[0]).toEqual({
    url: "/api/v1/reviews/cml/topic%231",
    body: { rating: 3, expected_due: card.due, expected_content_hash: card.content_hash },
  });
  await act(async () =>
    pending.resolve(
      jsonResponse({
        card_id: card.card_id,
        topic_id: card.topic_id,
        rating: 3,
        due: "2026-10-26T00:00:00Z",
        mastery: "introduced",
      }),
    ),
  );
  expect(await screen.findByText("This review batch is complete.")).toBeVisible();
});

it("keeps the revealed answer after a write failure and allows retry", async () => {
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) =>
    init.method === "POST"
      ? failure(500, "Rating not saved")
      : jsonResponse({ enrolled: 0, cards: [card] }),
  );
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "1 Again" }));
  expect(await screen.findByText("Rating not saved")).toBeVisible();
  expect(screen.getByText("a vector")).toBeVisible();
  expect(screen.getByRole("button", { name: "1 Again" })).toBeEnabled();
});

it("preserves the old answer on 409, refreshes, and requires revealing changed content before rating", async () => {
  let loads = 0;
  const writes: unknown[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") {
      writes.push(JSON.parse(String(init.body)));
      return writes.length === 1
        ? failure(409, "Card changed")
        : jsonResponse({
            card_id: card.card_id,
            topic_id: card.topic_id,
            rating: 4,
            mastery: "introduced",
            due: "2026-10-26T00:00:00Z",
          });
    }
    loads += 1;
    return jsonResponse({
      enrolled: 0,
      cards: [
        loads === 1
          ? card
          : {
              ...card,
              question: "Updated question",
              answer: "Updated answer",
              content_hash: "fresh-hash",
              due: "2026-10-25T01:00:00Z",
            },
      ],
    });
  });
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "4 Easy" }));
  await waitFor(() => expect(loads).toBe(2));
  expect(screen.getByText("a vector")).toBeVisible();
  expect(screen.queryByText("Updated question")).not.toBeInTheDocument();
  fireEvent.keyDown(window, { key: "4" });
  expect(writes).toHaveLength(1);
  fireEvent.click(await screen.findByRole("button", { name: "Use updated queue" }));
  expect(await screen.findByText("Updated question")).toBeVisible();
  expect(screen.queryByText("Updated answer")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "4 Easy" }));
  await waitFor(() => expect(writes).toHaveLength(2));
  expect(writes[1]).toEqual({
    rating: 4,
    expected_due: "2026-10-25T01:00:00Z",
    expected_content_hash: "fresh-hash",
  });
});

it("loads a bounded queue and shows an actionable empty state", async () => {
  const fetcher = vi.fn(async (_url: string) => jsonResponse({ enrolled: 0, cards: [] }));
  vi.stubGlobal("fetch", fetcher);
  page(<ReviewPage />);
  expect(await screen.findByText("Nothing is due right now.")).toBeVisible();
  expect(fetcher.mock.calls[0]?.[0]).toBe("/api/v1/reviews/due?limit=20");
  expect(screen.getByRole("link", { name: "Back to Today" })).toHaveAttribute("href", "/today");
  expect(screen.queryByRole("button", { name: "Reveal answer" })).not.toBeInTheDocument();
});

it("shows review load failure with a retry action", async () => {
  let loads = 0;
  vi.stubGlobal("fetch", async () =>
    ++loads === 1
      ? failure(500, "Queue unavailable")
      : jsonResponse({ enrolled: 0, cards: [card] }),
  );
  page(<ReviewPage />);
  expect(await screen.findByText("Queue unavailable")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByRole("button", { name: "Reveal answer" })).toBeVisible();
});

it("keeps the previous answer if refreshing a stale card fails, then recovers an empty queue", async () => {
  let loads = 0;
  let writes = 0;
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") {
      writes += 1;
      return failure(409, "Schedule changed");
    }
    loads += 1;
    if (loads === 1) return jsonResponse({ enrolled: 0, cards: [card] });
    if (loads === 2) return failure(500, "Refresh unavailable");
    return jsonResponse({ enrolled: 0, cards: [] });
  });
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  expect(await screen.findByText("Refresh unavailable")).toBeVisible();
  expect(screen.getByText("a vector")).toBeVisible();
  expect(screen.getByRole("button", { name: "3 Good" })).toBeDisabled();
  fireEvent.keyDown(window, { key: "3" });
  expect(writes).toBe(1);
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  fireEvent.click(await screen.findByRole("button", { name: "Use updated queue" }));
  expect(await screen.findByText("Nothing is due right now.")).toBeVisible();
});

it("hides the answer again when advancing to a different card", async () => {
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) =>
    init.method === "POST"
      ? jsonResponse({
          card_id: card.card_id,
          topic_id: card.topic_id,
          due: "2026-10-26T00:00:00Z",
          rating: 2,
          mastery: "introduced",
        })
      : jsonResponse({
          enrolled: 0,
          cards: [
            card,
            {
              ...card,
              card_id: "cml/topic#2",
              question: "Second question",
              answer: "Second answer",
            },
          ],
        }),
  );
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "2 Hard" }));
  expect(await screen.findByText("Second question")).toBeVisible();
  expect(screen.queryByText("Second answer")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Reveal answer" })).toBeVisible();
});

it("prevents energy replanning during a pending close and blocks duplicate close writes", async () => {
  const pending = deferred<Response>();
  let writes = 0;
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") {
      writes += 1;
      return pending.promise;
    }
    return jsonResponse(today);
  });
  page(<TodayPage />);
  await screen.findByText("Learn · 45 min");
  fireEvent.change(screen.getByLabelText("Where I stopped"), { target: { value: "stopped" } });
  fireEvent.change(screen.getByLabelText("Exact next step"), { target: { value: "continue" } });
  fireEvent.click(screen.getByRole("button", { name: "Save note and close" }));
  fireEvent.submit(screen.getByLabelText("Where I stopped").closest("form") as HTMLFormElement);
  await waitFor(() => expect(writes).toBe(1));
  expect(screen.getByRole("button", { name: "Low energy" })).toBeDisabled();
  await act(async () => pending.resolve(failure(500, "Close unavailable")));
  expect(await screen.findByText("Close unavailable")).toBeVisible();
  expect(screen.getByRole("button", { name: "Low energy" })).toBeEnabled();
});

it("offers the real core problem after a review first action", async () => {
  vi.stubGlobal("fetch", async () =>
    jsonResponse({
      ...today,
      current: { ...today.current, first_action: "Run dogma review --limit 5." },
    }),
  );
  page(<TodayPage />);
  expect(await screen.findByRole("link", { name: "Open core problem" })).toHaveAttribute(
    "href",
    "/problem/cml/topic/p01",
  );
});

it("links the familiar re-solve after a review first action without inventing a must-do", async () => {
  vi.stubGlobal("fetch", async () =>
    jsonResponse({
      ...today,
      current: {
        ...today.current,
        kind: "consolidate",
        must_do: null,
        first_action: "Run dogma review --limit 15.",
        blocks: [
          { name: "Review", minutes: 15, detail: "Review cards" },
          {
            name: "Re-solve",
            minutes: 25,
            detail:
              "Re-solve Familiar (cml/topic/familiar): dogma open cml/topic/familiar, then dogma run cml/topic/familiar.",
          },
          { name: "Close", minutes: 5, detail: "Leave a note" },
        ],
      },
    }),
  );
  page(<TodayPage />);
  expect(await screen.findByRole("link", { name: "Open familiar problem" })).toHaveAttribute(
    "href",
    "/problem/cml/topic/familiar",
  );
  expect(screen.queryByText(/One must-do/)).not.toBeInTheDocument();
});

it("requires another reveal if a refreshed queue replaces the visible card snapshot", async () => {
  let loads = 0;
  vi.stubGlobal("fetch", async () =>
    jsonResponse({
      enrolled: 0,
      cards: [
        ++loads === 1 ? card : { ...card, content_hash: "new-hash", answer: "Changed answer" },
      ],
    }),
  );
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <MemoryRouter>
      <QueryClientProvider client={client}>
        <ReviewPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  expect(screen.getByText("a vector")).toBeVisible();
  await act(async () => {
    await client.invalidateQueries({ queryKey: ["reviews"] });
  });
  expect(await screen.findByRole("button", { name: "Reveal answer" })).toBeVisible();
  expect(screen.queryByText("Changed answer")).not.toBeInTheDocument();
});

it("presents planner instructions as browser actions while retaining the detailed learning goals", async () => {
  const session = today.current;
  if (!session) throw new Error("Synthetic session is required");
  vi.stubGlobal("fetch", async () =>
    jsonResponse({
      ...today,
      current: {
        ...session,
        notes: ["Nothing is due. Finish with dogma close, then rest. No new material is planned."],
        blocks: [
          {
            name: "Retrieve",
            minutes: 5,
            detail: "Answer up to 5 due cards (7 due now): dogma review --limit 5",
          },
          {
            name: "Predict",
            minutes: 3,
            detail:
              "Before you read content/cml/topic/lesson.md, write one guess for this goal: explain why the gradient points uphill.",
          },
          {
            name: "Learn",
            minutes: 12,
            detail:
              "Read content/cml/topic/lesson.md (the system, the worked example, the interview angle).",
          },
          {
            name: "Practice",
            minutes: 20,
            detail:
              "Must-do: dogma open cml/topic/p01, read example 1, then solve it. Check public tests with dogma run cml/topic/p01, then dogma submit cml/topic/p01 to record a successful solve.",
          },
          {
            name: "Close",
            minutes: 5,
            detail:
              "dogma close: where you stopped, the exact next step, any blocker, and a 60-second answer in your own words.",
          },
        ],
      },
    }),
  );
  page(<TodayPage />);
  const plan = await screen.findByRole("region", { name: "Learn · 45 min" });
  expect(plan).toHaveTextContent(
    "write one guess for this goal: explain why the gradient points uphill.",
  );
  expect(plan).toHaveTextContent("the system, the worked example, the interview angle");
  expect(plan).toHaveTextContent("read example 1, then solve it");
  expect(plan).toHaveTextContent(
    "Check public tests with Run, then Submit to record a successful solve.",
  );
  expect(plan).toHaveTextContent(
    "the exact next step, any blocker, and a 60-second answer in your own words",
  );
  expect(plan).toHaveTextContent("Write your Predict guess (3 min), then open the Lesson tab.");
  expect(plan).toHaveTextContent("No new material is planned.");
  expect(plan).toHaveTextContent("Review page");
  expect(plan).toHaveTextContent("close form below");
  expect(plan).not.toHaveTextContent(
    /\bdogma\s+(?:open|run|submit|review|close)\b|content\/cml\/topic\/lesson\.md|--limit/,
  );
  expect(within(plan).getByRole("link", { name: "Start session" })).toHaveAttribute(
    "href",
    "/problem/cml/topic/p01",
  );
  expect(
    within(plan)
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getByRole("rowheader").textContent),
  ).toEqual(["Retrieve", "Predict", "Learn", "Practice", "Close"]);
  expect(
    within(plan)
      .getAllByRole("cell")
      .filter((cell) => /min$/.test(cell.textContent ?? ""))
      .map((cell) => cell.textContent),
  ).toEqual(["5 min", "3 min", "12 min", "20 min", "5 min"]);
});

const second: components["schemas"]["ReviewCardOut"] = {
  ...card,
  card_id: "cml/second#1",
  topic_id: "cml/second",
  question: "Second question",
  answer: "Second answer",
  content_hash: "second-hash",
};
function ratedOk(snapshot: components["schemas"]["ReviewCardOut"], rating: number) {
  return jsonResponse({
    card_id: snapshot.card_id,
    topic_id: snapshot.topic_id,
    rating,
    mastery: "introduced",
    due: "2026-10-26T00:00:00Z",
  });
}
function coded(status: number, code: string, detail: string) {
  return jsonResponse(
    { type: "about:blank", title: "Failed", status, detail, code, instance: "", errors: null },
    status,
  );
}
it("treats a card that disappeared before rating like a stale card, without a duplicate write", async () => {
  let loads = 0;
  const writes: unknown[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") {
      writes.push(JSON.parse(String(init.body)));
      return coded(404, "REVIEW_NOT_FOUND", "This card is no longer available.");
    }
    loads += 1;
    return jsonResponse({
      enrolled: 0,
      cards: [
        loads === 1 ? card : { ...card, question: "Refreshed question", content_hash: "fresh" },
      ],
    });
  });
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  expect(await screen.findByText(/schedule or content changed/)).toBeVisible();
  await waitFor(() => expect(loads).toBe(2));
  expect(screen.getByText("a vector")).toBeVisible();
  fireEvent.keyDown(window, { key: "4" });
  expect(writes).toHaveLength(1);
  fireEvent.click(await screen.findByRole("button", { name: "Use updated queue" }));
  expect(await screen.findByText("Refreshed question")).toBeVisible();
  expect(screen.queryByText("a vector")).not.toBeInTheDocument();
  expect(writes).toHaveLength(1);
});

it("moves focus to the answer region when the answer is revealed", async () => {
  vi.stubGlobal("fetch", async () => jsonResponse({ enrolled: 0, cards: [card] }));
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  expect(screen.getByRole("region", { name: "Answer" })).toHaveFocus();
});

it("moves focus to the next card heading after a successful rating", async () => {
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) =>
    init.method === "POST"
      ? ratedOk(card, 3)
      : jsonResponse({ enrolled: 0, cards: [card, second] }),
  );
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  expect(await screen.findByText("Second question")).toBeVisible();
  await waitFor(() => expect(screen.getByRole("heading", { name: "cml/second" })).toHaveFocus());
});

it("moves focus to the batch-complete heading after the last rating", async () => {
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) =>
    init.method === "POST" ? ratedOk(card, 3) : jsonResponse({ enrolled: 0, cards: [card] }),
  );
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  const complete = await screen.findByRole("heading", { name: "This review batch is complete." });
  await waitFor(() => expect(complete).toHaveFocus());
});

it("moves focus to the refreshed card heading after using the updated queue", async () => {
  let loads = 0;
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") return coded(409, "REVIEW_STALE", "Changed");
    loads += 1;
    return jsonResponse({
      enrolled: 0,
      cards: [loads === 1 ? card : { ...card, question: "Refreshed question", content_hash: "f" }],
    });
  });
  page(<ReviewPage />);
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  fireEvent.click(await screen.findByRole("button", { name: "Use updated queue" }));
  await waitFor(() => expect(screen.getByRole("heading", { name: "cml/topic" })).toHaveFocus());
});

it("keeps the batch size while rating and starts a new batch count after it finishes", async () => {
  let loads = 0;
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    if (init.method === "POST") return ratedOk(card, 3);
    loads += 1;
    if (loads === 1) return jsonResponse({ enrolled: 0, cards: [card, second] });
    return jsonResponse({
      enrolled: 0,
      cards: [1, 2, 3].map((n) => ({
        ...card,
        card_id: `cml/next#${n}`,
        topic_id: `cml/next${n}`,
      })),
    });
  });
  page(<ReviewPage />);
  expect(await screen.findByText("2 cards in this batch · 0 reviewed")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  expect(await screen.findByText("2 cards in this batch · 1 reviewed")).toBeVisible();
  fireEvent.click(await screen.findByRole("button", { name: "Reveal answer" }));
  fireEvent.click(screen.getByRole("button", { name: "3 Good" }));
  expect(
    await screen.findByRole("heading", { name: "This review batch is complete." }),
  ).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Check for due cards" }));
  expect(await screen.findByText("3 cards in this batch · 0 reviewed")).toBeVisible();
});
