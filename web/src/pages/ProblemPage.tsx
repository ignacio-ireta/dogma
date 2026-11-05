// The Problem page (ARCHITECTURE §4.8): material on the left, the editor on the right, results
// under the editor. The page is keyed by problem id, so each problem starts with fresh state.
import { useRef, useState } from "react";
import { useParams } from "react-router";
import { describeError } from "../api/client";
import type { RunMode } from "../api/types";
import { AnswerPanel } from "../components/answer/AnswerPanel";
import { CodeEditor } from "../components/CodeEditor";
import { EditorToolbar } from "../components/EditorToolbar";
import { HintsPanel } from "../components/HintsPanel";
import { StudyIllustration } from "../components/illustrations/StudyIllustration";
import { LessonPanel } from "../components/LessonPanel";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { ResolveWindowPanel } from "../components/ResolveWindowPanel";
import { ResultsPanel } from "../components/ResultsPanel";
import type { ShortcutItem } from "../components/ShortcutHelp";
import { StatementPanel } from "../components/StatementPanel";
import { type TabDef, Tabs } from "../components/Tabs";
import { TourStrip } from "../components/TourStrip";
import { useAttemptRun } from "../hooks/useAttemptRun";
import { useCurrentResolve } from "../hooks/useCurrentResolve";
import { useProblem } from "../hooks/useProblem";
import { useShortcuts } from "../hooks/useShortcuts";
import { useTopic } from "../hooks/useTopic";
import { useWorkspaceDraft } from "../hooks/useWorkspaceDraft";
import type { ShortcutBinding } from "../lib/shortcuts";
import { NotFoundPage } from "./NotFoundPage";

type PanelId = "statement" | "lesson" | "answer" | "hints";

const TABS: readonly TabDef<PanelId>[] = [
  { id: "statement", label: "Statement" },
  { id: "lesson", label: "Lesson" },
  { id: "answer", label: "Answer" },
  { id: "hints", label: "Hints" },
];

export const SHORTCUT_HELP: readonly ShortcutItem[] = [
  { keys: "Cmd/Ctrl + Enter", label: "Run your file against the public tests" },
  { keys: "Cmd/Ctrl + Shift + Enter", label: "Submit: public and hidden tests, recorded" },
  { keys: "Cmd/Ctrl + H", label: "Next hint (level 1, then 2, then 3)" },
];

export function ProblemPage() {
  const { track, topic, problem } = useParams();
  if (track === undefined || topic === undefined || problem === undefined) {
    return <NotFoundPage />;
  }
  const problemId = `${track}/${topic}/${problem}`;
  return <ProblemWorkspace key={problemId} problemId={problemId} />;
}

function ProblemWorkspace({ problemId }: { readonly problemId: string }) {
  const topicId = problemId.split("/").slice(0, 2).join("/");
  const problemQuery = useProblem(problemId);
  const topicQuery = useTopic(topicId);
  const draft = useWorkspaceDraft(problemId, problemQuery.data?.starter);
  const attempt = useAttemptRun(problemId, draft.saveNow);
  const openWindow = useCurrentResolve();
  const timedWindow =
    openWindow.data?.problem_id === problemId && openWindow.data.status === "open"
      ? openWindow.data
      : null;
  const [panel, setPanel] = useState<PanelId>("statement");
  // The Answer tab loads its questions only once it is opened, then keeps its draft while hidden.
  const [answerOpened, setAnswerOpened] = useState(false);
  const selectPanel = (next: PanelId) => {
    if (next === "answer") setAnswerOpened(true);
    setPanel(next);
  };
  const nextHint = useRef<() => void>(() => undefined);

  const runnable = draft.loaded && draft.status.kind !== "too-large" && !attempt.isPending;
  const runMode = (mode: RunMode) => {
    if (runnable) {
      attempt.mutate(mode);
    }
  };

  const bindings: readonly ShortcutBinding[] = [
    { id: "run", key: "Mod-Enter", label: "Run", run: () => runMode("run") },
    { id: "submit", key: "Mod-Shift-Enter", label: "Submit", run: () => runMode("submit") },
    { id: "hint", key: "Mod-h", label: "Next hint", run: () => nextHint.current() },
  ];
  useShortcuts(bindings);

  const running = attempt.isPending ? (attempt.variables ?? null) : null;
  const problemData = problemQuery.data;
  const fileName =
    problemData === undefined
      ? ""
      : `${problemId.split("/").pop() ?? problemId}.${problemData.language === "sql" ? "sql" : "py"}`;

  if (problemQuery.isError) {
    return <Notice tone="error">{describeError(problemQuery.error)}</Notice>;
  }

  return (
    <div className="problem">
      <header className="problem__header">
        <div className="problem__heading-copy">
          <p className="eyebrow">{topicQuery.data?.title ?? topicId}</p>
          <h1>{problemData?.title ?? "Problem"}</h1>
          {problemData !== undefined ? (
            <p className="muted">
              {problemData.minutes} min · {problemData.language} · runtime {problemData.runtime}
            </p>
          ) : null}
        </div>
        <div className="problem__illustration">
          <StudyIllustration topicId={topicId} compact />
        </div>
      </header>

      {draft.loadError !== null && draft.loadError !== undefined ? (
        <Notice tone="error">{describeError(draft.loadError)}</Notice>
      ) : null}

      {timedWindow !== null ? <ResolveWindowPanel window={timedWindow} /> : null}

      <TourStrip />

      <div className="problem__layout">
        <section className="problem__material" aria-label="Problem material">
          <Tabs
            label="Problem material"
            tabs={TABS}
            active={panel}
            onSelect={selectPanel}
            idPrefix="problem"
            panels={{
              statement: <StatementPanel markdown={problemData?.problem_md} />,
              lesson: topicQuery.isError ? (
                <Notice tone="error">{describeError(topicQuery.error)}</Notice>
              ) : (
                <LessonPanel topic={topicQuery.data} />
              ),
              answer: answerOpened ? <AnswerPanel topicId={topicId} /> : null,
              hints:
                problemData === undefined ? (
                  <Loading what="hints" />
                ) : (
                  <HintsPanel
                    problemId={problemId}
                    hintsCount={problemData.hints_count}
                    timedWindowOpen={timedWindow !== null}
                    onRegister={(next) => {
                      nextHint.current = next;
                    }}
                  />
                ),
            }}
          />
        </section>

        <section className="problem__workbench" aria-label="Your solution">
          <EditorToolbar
            fileName={fileName}
            status={draft.status}
            running={running}
            disabled={!runnable}
            onRun={() => runMode("run")}
            onSubmit={() => runMode("submit")}
            shortcuts={SHORTCUT_HELP}
          />
          {draft.loaded && problemData !== undefined ? (
            <CodeEditor
              value={draft.text}
              language={problemData.language}
              label={`Your ${problemData.language} solution`}
              bindings={bindings}
              onChange={draft.onChange}
            />
          ) : draft.loadError == null ? (
            <Loading what="your file" />
          ) : null}
          <ResultsPanel result={attempt.data} error={attempt.error} running={running} />
        </section>
      </div>
    </div>
  );
}
