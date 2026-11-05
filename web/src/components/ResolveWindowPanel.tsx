// The open timed re-solve on its Problem page: budget, a calm countdown, the rules and Abandon.
// Past the budget the wording changes, but nothing turns red and nothing interrupts the work.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { describeError } from "../api/client";
import { abandonResolve } from "../api/resolves";
import type { ResolveWindowOut } from "../api/types";
import { useNow } from "../hooks/useNow";
import { clockText, remainingSeconds } from "../lib/resolves";
import { ConfirmDialog } from "./ConfirmDialog";
import { Notice } from "./Notice";
import "../styles/resolves.css";

export function ResolveWindowPanel({ window }: { readonly window: ResolveWindowOut }) {
  const now = useNow(true);
  const client = useQueryClient();
  const [confirming, setConfirming] = useState(false);
  const abandon = useMutation({
    mutationFn: abandonResolve,
    onSuccess: () => client.invalidateQueries({ queryKey: ["resolve"] }),
  });
  const left = remainingSeconds(window.deadline, now);
  const over = left === 0;
  return (
    <section className={`resolve${over ? " resolve--over" : ""}`} aria-labelledby="resolve-title">
      <h2 id="resolve-title" className="resolve__title">
        Timed re-solve open
      </h2>
      <p className="resolve__budget">Time budget {Math.floor(window.budget_s / 60)} min</p>
      <p className="resolve__left">{over ? "The budget has passed." : `${clockText(left)} left`}</p>
      {over ? (
        <p className="muted">Submit when you are ready. A late pass counts as partial.</p>
      ) : null}
      <ul className="resolve__rules">
        <li>Run is unlimited. Run as often as you like.</li>
        <li>The first submit decides it: passed, partial or failed.</li>
        <li>Hints count as practice, not a timed pass.</li>
      </ul>
      <p className="muted">Hints used: {window.hints_used}</p>
      {abandon.isError ? <Notice tone="error">{describeError(abandon.error)}</Notice> : null}
      {confirming ? (
        <ConfirmDialog
          title="Abandon this timed re-solve?"
          body="No penalty: the schedule does not change. You can start it again later."
          confirmLabel="Yes, abandon"
          cancelLabel="Keep going"
          onCancel={() => setConfirming(false)}
          onConfirm={() => {
            setConfirming(false);
            abandon.mutate();
          }}
        />
      ) : (
        <button type="button" className="button" onClick={() => setConfirming(true)}>
          Abandon re-solve
        </button>
      )}
    </section>
  );
}
