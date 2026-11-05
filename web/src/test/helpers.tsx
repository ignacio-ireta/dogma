import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { PropsWithChildren } from "react";
import type { AttemptCreated } from "../api/types";

export function queryWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

export function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export function deferred<T>() {
  let resolve: (value: T) => void = () => {};
  let reject: (reason: unknown) => void = () => {};
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

export const attemptResult: AttemptCreated = {
  mode: "submit",
  recorded: true,
  record_warning: null,
  timed_resolve: null,
  report: {
    attempt_id: "scratch-attempt",
    problem_id: "cml/topic/problem",
    problem_version: 1,
    runtime: "python",
    duration_ms: 10,
    status: "failed",
    static: [],
    findings: [],
    headline: null,
    message: "",
    stdout: "",
    tests: [
      {
        file: "public",
        name: "test_public",
        outcome: "fail",
        duration_ms: 1,
        message: "assert [1, 2] == [1, 3]\nFull public detail",
      },
      {
        file: "hidden",
        name: "test_hidden",
        outcome: "fail",
        duration_ms: 1,
        message: "Assertion failed (details hidden)",
      },
    ],
  },
};
