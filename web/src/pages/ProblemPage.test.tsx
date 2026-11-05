import { EditorView } from "@codemirror/view";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Link, MemoryRouter, Route, Routes } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import type { ProblemOut, TopicOut } from "../api/types";
import { deferred, jsonResponse, queryWrapper } from "../test/helpers";
import { ProblemPage } from "./ProblemPage";

const id = "cml/topic/problem";
const problem: ProblemOut = {
  id,
  title: "Scratch problem",
  track: "cml",
  topic_id: "cml/topic",
  version: 1,
  status: "verified",
  language: "python",
  runtime: "py-numpy",
  minutes: 15,
  difficulty: "easy",
  entry: "solve",
  hints_count: 0,
  banned_imports: [],
  core: true,
  problem_md: "Scratch statement",
  starter: "def solve():\n    pass",
  timeout_s: 30,
  tolerance: { rtol: 0.00001, atol: 0.00001 },
};
const topic: TopicOut = {
  id: "cml/topic",
  title: "Scratch topic",
  track: "cml",
  aliases: [],
  cards_count: 0,
  interview_angle: "Scratch",
  interview_present: false,
  lesson: null,
  minutes: { lesson: 15, min_viable: 15, practice: 15 },
  objectives: [],
  problems: [],
  requires: [],
  sources: [],
  status: "verified",
  tier: "A",
};
const failure = (detail: string) =>
  jsonResponse(
    {
      type: "about:blank",
      title: "Failed",
      status: 404,
      code: "API_NOT_FOUND",
      detail,
      instance: "",
      errors: null,
    },
    404,
  );

function page() {
  return render(
    <MemoryRouter initialEntries={[`/problem/${id}`]}>
      <Routes>
        <Route path="/problem/:track/:topic/:problem" element={<ProblemPage />} />
      </Routes>
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
}

afterEach(() => vi.unstubAllGlobals());

it("loads a client route with Run disabled until its workspace has arrived", async () => {
  const workspace = deferred<Response>();
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.includes("/workspace/")) return workspace.promise;
    return jsonResponse(url.includes("/topics/") ? topic : problem);
  });
  page();
  expect(screen.getByRole("button", { name: /^Run/ })).toBeDisabled();
  await screen.findByRole("heading", { name: "Scratch problem" });
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  await act(async () =>
    workspace.resolve(jsonResponse({ problem_id: id, exists: true, content: "# saved solution" })),
  );
  await waitFor(() => expect(screen.getByRole("button", { name: /^Run/ })).toBeEnabled());
  // The CodeMirror view mounts in a passive effect, so the textbox can appear after Run is enabled.
  await waitFor(() => expect(screen.getByRole("textbox")).toHaveTextContent("# saved solution"));
});

it("shows a failed workspace load and never enables grading", async () => {
  vi.stubGlobal("fetch", async (url: string) =>
    url.includes("/workspace/")
      ? failure("Could not load your scratch file")
      : jsonResponse(url.includes("/topics/") ? topic : problem),
  );
  page();
  expect(await screen.findByText("Could not load your scratch file")).toBeVisible();
  expect(screen.getByRole("button", { name: /^Run/ })).toBeDisabled();
  expect(screen.queryByText(/Loading your file/)).not.toBeInTheDocument();
});

it("shows a topic error in the lesson panel instead of loading forever", async () => {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.includes("/topics/")) return failure("Could not load the scratch topic");
    return jsonResponse(
      url.includes("/workspace/") ? { problem_id: id, exists: true, content: "# saved" } : problem,
    );
  });
  page();
  fireEvent.click(screen.getByRole("tab", { name: "Lesson" }));
  expect(await screen.findByText("Could not load the scratch topic")).toBeVisible();
  expect(screen.queryByText("Loading the lesson…")).not.toBeInTheDocument();
});

it("shows a problem error without indefinite statement and editor loading", async () => {
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.includes("/problems/")) return failure("Scratch problem was not found");
    return jsonResponse(
      url.includes("/workspace/") ? { problem_id: id, exists: true, content: "# saved" } : topic,
    );
  });
  page();
  expect(await screen.findByText("Scratch problem was not found")).toBeVisible();
  expect(screen.queryByText("Loading the statement…")).not.toBeInTheDocument();
});

it("keeps a late workspace response from a previous route out of the new editor", async () => {
  const oldWorkspace = deferred<Response>();
  const nextId = "cml/topic/next-problem";
  vi.stubGlobal("fetch", async (url: string) => {
    if (url.endsWith(`/workspace/${id}`)) return oldWorkspace.promise;
    if (url.includes("/workspace/"))
      return jsonResponse({ problem_id: nextId, exists: true, content: "# next saved file" });
    if (url.includes("/topics/")) return jsonResponse(topic);
    return jsonResponse(
      url.endsWith("next-problem")
        ? { ...problem, id: nextId, title: "Next scratch problem" }
        : problem,
    );
  });
  render(
    <MemoryRouter initialEntries={[`/problem/${id}`]}>
      <Link to={`/problem/${nextId}`}>Next problem</Link>
      <Routes>
        <Route path="/problem/:track/:topic/:problem" element={<ProblemPage />} />
      </Routes>
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
  await screen.findByRole("heading", { name: "Scratch problem" });
  fireEvent.click(screen.getByRole("link", { name: "Next problem" }));
  await waitFor(() => expect(screen.getByRole("textbox")).toHaveTextContent("# next saved file"));
  await act(async () =>
    oldWorkspace.resolve(
      jsonResponse({ problem_id: id, exists: true, content: "# old late file" }),
    ),
  );
  expect(screen.getByRole("heading", { name: "Next scratch problem" })).toBeVisible();
  expect(screen.getByRole("textbox")).toHaveTextContent("# next saved file");
});

it("restores the editor and visible save error after navigating away from a failed save", async () => {
  const departure = deferred<Response>();
  let writes = 0;
  let disk = "# old disk file";
  vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
    if (init.method === "PUT") {
      writes += 1;
      if (writes === 1) return departure.promise;
      disk = (JSON.parse(String(init.body)) as { content: string }).content;
    }
    if (url.includes("/workspace/"))
      return jsonResponse({ problem_id: id, exists: true, content: disk });
    return jsonResponse(url.includes("/topics/") ? topic : problem);
  });
  render(
    <MemoryRouter initialEntries={[`/problem/${id}`]}>
      <Link to="/away">Leave problem</Link>
      <Link to={`/problem/${id}`}>Return to problem</Link>
      <Routes>
        <Route path="/problem/:track/:topic/:problem" element={<ProblemPage />} />
        <Route path="/away" element={<p>Away</p>} />
      </Routes>
    </MemoryRouter>,
    { wrapper: queryWrapper() },
  );
  const editor = await screen.findByRole("textbox");
  const view = EditorView.findFromDOM(editor);
  expect(view).not.toBeNull();
  act(() =>
    view?.dispatch({
      changes: { from: 0, to: view.state.doc.length, insert: "# keep this unsaved solution" },
    }),
  );
  fireEvent.click(screen.getByRole("link", { name: "Leave problem" }));
  await waitFor(() => expect(writes).toBe(1));
  await act(async () =>
    departure.resolve(
      jsonResponse(
        {
          type: "about:blank",
          title: "Write failed",
          status: 500,
          code: "API_WORKSPACE_WRITE_FAILED",
          detail: "Scratch save rejected",
          instance: "",
          errors: null,
        },
        500,
      ),
    ),
  );
  fireEvent.click(screen.getByRole("link", { name: "Return to problem" }));
  await waitFor(() =>
    expect(screen.getByRole("textbox")).toHaveTextContent("# keep this unsaved solution"),
  );
  expect(screen.getByText("Not saved: Scratch save rejected")).toBeVisible();
  // A real editor edit and the normal debounce retry make the recovered buffer durable.
  await waitFor(() => expect(screen.getByText("Saved")).toBeVisible(), { timeout: 1500 });
  expect(disk).toBe("# keep this unsaved solution");
});

it("shows the open timed re-solve for this problem and asks before a hint counts against it", async () => {
  const now = Date.now();
  const window = {
    id: "scratch-window",
    problem_id: id,
    status: "open",
    started_at: new Date(now).toISOString(),
    budget_s: 1200,
    deadline: new Date(now + 1_200_000).toISOString(),
    hints_used: 0,
    closed_at: null,
  };
  const requests: string[] = [];
  vi.stubGlobal("fetch", async (url: string, init?: RequestInit) => {
    requests.push(`${init?.method ?? "GET"} ${url}`);
    if (url.endsWith("/resolves/current")) return jsonResponse(window);
    if (url.includes("/workspace/"))
      return jsonResponse({ problem_id: id, exists: true, content: "# saved" });
    if (url.includes("/topics/")) return jsonResponse(topic);
    return jsonResponse({ ...problem, hints_count: 3 });
  });
  page();
  expect(await screen.findByRole("heading", { name: "Timed re-solve open" })).toBeVisible();
  expect(screen.getByText(/Time budget 20 min/)).toBeVisible();
  fireEvent.click(screen.getByRole("tab", { name: "Hints" }));
  fireEvent.click(screen.getByRole("button", { name: "Hint 1: Concept" }));
  expect(
    screen.getByRole("alertdialog", { name: "Show a hint during the re-solve?" }),
  ).toBeVisible();
  expect(requests.some((entry) => entry.includes("/tutor/hint"))).toBe(false);
});
