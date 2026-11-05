// Nothing is open: start a mock, or read the debrief of the last scored one.
import { describeError } from "../../api/client";
import type { MockReportOut } from "../../api/types";
import { Notice } from "../../components/Notice";
import type { useMockActions } from "../../hooks/useMock";

export function StartPanel({
  actions,
  lastScored,
  onViewDebrief,
}: {
  readonly actions: ReturnType<typeof useMockActions>;
  readonly lastScored: MockReportOut | null;
  readonly onViewDebrief: (sessionId: string) => void;
}) {
  return (
    <section className="mock-panel" aria-labelledby="start-mock-title">
      <h2 id="start-mock-title">Start a mock</h2>
      <p>
        The planner picks a case, a coding problem you have passed and a theory question from a
        topic you have retained. The part timings are a plan, not a deadline.
      </p>
      {actions.start.isError && <Notice tone="error">{describeError(actions.start.error)}</Notice>}
      <button
        className="button button--primary"
        type="button"
        disabled={actions.start.isPending}
        onClick={() => actions.start.mutate(undefined)}
      >
        Start a mock
      </button>
      {lastScored && (
        <p className="mock-panel__last">
          <button
            className="button"
            type="button"
            onClick={() => onViewDebrief(lastScored.session_id)}
          >
            Read the debrief of your last mock
          </button>
        </p>
      )}
    </section>
  );
}
