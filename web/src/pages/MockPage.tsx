// The Mock page (ARCHITECTURE §4.4, route /mock): start a timed mock, work through its plan, score
// the 30 items, then read the debrief. The server's open-mock status decides which step shows.
import { useState } from "react";
import { describeError } from "../api/client";
import { Loading } from "../components/Loading";
import { Notice } from "../components/Notice";
import { PageHeading } from "../components/PageHeading";
import { useMockActions, useMockStatus } from "../hooks/useMock";
import "../styles/interview.css";
import { DebriefView } from "./mock/DebriefView";
import { OpenMock } from "./mock/OpenMock";
import { StartPanel } from "./mock/StartPanel";

export function MockPage() {
  const status = useMockStatus();
  const actions = useMockActions();
  const [debriefId, setDebriefId] = useState<string | null>(null);
  const data = status.data;
  return (
    <div className="page mock-page">
      <PageHeading
        title="Mock interview"
        eyebrow="A calm run-through"
        description="A coding problem, a theory question and a case, then the 30-item rubric. Scores are yours."
      />
      {status.isPending && <Loading what="your mock" />}
      {status.isError && (
        <>
          <Notice tone="error">{describeError(status.error)}</Notice>
          <button className="button" type="button" onClick={() => void status.refetch()}>
            Try again
          </button>
        </>
      )}
      {data !== undefined &&
        (debriefId !== null ? (
          <DebriefView sessionId={debriefId} onBack={() => setDebriefId(null)} />
        ) : data.open ? (
          <OpenMock
            plan={data.open}
            actions={actions}
            onScored={(sessionId) => setDebriefId(sessionId)}
          />
        ) : (
          <StartPanel
            actions={actions}
            lastScored={data.last_scored}
            onViewDebrief={(sessionId) => setDebriefId(sessionId)}
          />
        ))}
    </div>
  );
}
