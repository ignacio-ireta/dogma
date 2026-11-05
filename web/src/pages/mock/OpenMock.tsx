// The open mock: its case brief and plan, the part timings, the question box, the score form and
// a calm way to abandon it. Nothing here counts down or turns red.
import { useState } from "react";
import { describeError } from "../../api/client";
import type { MockPlanOut } from "../../api/types";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { Markdown } from "../../components/Markdown";
import { Notice } from "../../components/Notice";
import type { useMockActions } from "../../hooks/useMock";
import { useNow } from "../../hooks/useNow";
import { elapsedLabel, percentText } from "../../lib/mock";
import { PartTimings } from "./PartTimings";
import { QuestionBox } from "./QuestionBox";
import { ScoreForm } from "./ScoreForm";

export function OpenMock({
  plan,
  actions,
  onScored,
}: {
  readonly plan: MockPlanOut;
  readonly actions: ReturnType<typeof useMockActions>;
  readonly onScored: (sessionId: string) => void;
}) {
  const now = useNow(true);
  const [confirming, setConfirming] = useState(false);
  return (
    <div className="mock-open">
      <p className="muted">
        Started {new Date(plan.started_at).toLocaleTimeString()} ·{" "}
        {elapsedLabel(plan.started_at, now)}
      </p>
      {plan.focus && (
        <Notice tone="info">
          Your focus: {plan.focus.title}, at {percentText(plan.focus.percent)} over your last three
          mocks.
        </Notice>
      )}
      <section className="mock-panel" aria-labelledby="mock-brief-title">
        <h2 id="mock-brief-title">The case</h2>
        {plan.case ? (
          <>
            <h3>{plan.case.title}</h3>
            {plan.case.brief ? (
              <Markdown source={plan.case.brief} />
            ) : (
              <p className="muted">This case has no brief yet.</p>
            )}
          </>
        ) : (
          <p className="muted">No case is planned for this mock.</p>
        )}
        {plan.coding && (
          <p>
            Live coding: <strong>{plan.coding.title}</strong>
          </p>
        )}
        {plan.theory && (
          <p>
            Theory: <strong>{plan.theory.title}</strong>
            {plan.theory.question_title ? `, ${plan.theory.question_title}` : null}
          </p>
        )}
      </section>
      <PartTimings parts={plan.parts} />
      <QuestionBox reveal={actions.reveal} />
      <ScoreForm actions={actions} onScored={onScored} />
      {actions.abandon.isError && (
        <Notice tone="error">{describeError(actions.abandon.error)}</Notice>
      )}
      {confirming ? (
        <ConfirmDialog
          title="Abandon this mock?"
          body="Nothing is stored for it, and no scores are kept. You can start a new one at once."
          confirmLabel="Abandon mock"
          cancelLabel="Keep going"
          onConfirm={() => {
            setConfirming(false);
            actions.abandon.mutate();
          }}
          onCancel={() => setConfirming(false)}
        />
      ) : (
        <button className="button" type="button" onClick={() => setConfirming(true)}>
          Abandon mock
        </button>
      )}
    </div>
  );
}
