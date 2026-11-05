// Timed re-solve calls (ARCHITECTURE §4.4). The window itself is read with getCurrentResolve; the
// verdict of a submit arrives on the attempt response, so no call here reports a result.
import { apiFetch } from "./client";
import type { ResolveStartedOut, ResolveWindowOut } from "./types";

export function startResolve(problemId: string): Promise<ResolveStartedOut> {
  return apiFetch<ResolveStartedOut>("/resolves", {
    method: "POST",
    body: JSON.stringify({ problem_id: problemId }),
  });
}

/** The open window, or null when none is open. The API answers 200 either way. */
export function getCurrentResolve(): Promise<ResolveWindowOut | null> {
  return apiFetch<ResolveWindowOut | null>("/resolves/current");
}

export function abandonResolve(): Promise<ResolveWindowOut> {
  return apiFetch<ResolveWindowOut>("/resolves/current/abandon", { method: "POST" });
}
