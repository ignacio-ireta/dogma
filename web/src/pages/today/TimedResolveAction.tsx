// The planned timed re-solve as Today's first action: start it, or continue it when its window is
// already open. Starting resets the workspace on the API side, so the Problem page is the next step.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router";
import { describeError, encodeId } from "../../api/client";
import { startResolve } from "../../api/resolves";
import type { TimedResolveOut } from "../../api/types";
import { Icon } from "../../components/Icon";
import { Notice } from "../../components/Notice";
import { CURRENT_RESOLVE_KEY, useCurrentResolve } from "../../hooks/useCurrentResolve";
import "../../styles/resolves.css";

export function TimedResolveAction({
  timed,
  disabled,
}: {
  readonly timed: TimedResolveOut;
  readonly disabled: boolean;
}) {
  const navigate = useNavigate();
  const client = useQueryClient();
  const current = useCurrentResolve();
  const problemPath = `/problem/${encodeId(timed.problem_id)}`;
  const start = useMutation({
    mutationFn: () => startResolve(timed.problem_id),
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: CURRENT_RESOLVE_KEY });
      navigate(problemPath);
    },
  });
  const windowOpen = current.data?.problem_id === timed.problem_id;

  return (
    <div className="resolve-action">
      {windowOpen ? (
        <Link className="button button--primary learning-start" to={problemPath}>
          Open timed re-solve
          <Icon name="arrow" />
        </Link>
      ) : (
        <button
          className="button button--primary learning-start"
          type="button"
          disabled={disabled || current.isPending || start.isPending}
          onClick={() => start.mutate()}
        >
          Start timed re-solve
          <Icon name="arrow" />
        </button>
      )}
      <p className="muted">
        {timed.title} · {timed.minutes} min, no hints. The first submit decides it.
      </p>
      {start.isError ? <Notice tone="error">{describeError(start.error)}</Notice> : null}
    </div>
  );
}
