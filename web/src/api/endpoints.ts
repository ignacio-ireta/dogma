// One function per call the browser makes. Hooks compose these; components never call fetch.
import { apiFetch, encodeId } from "./client";
import type {
  AttemptCreated,
  CatalogOut,
  ProblemOut,
  ProgressOut,
  RunMode,
  TopicOut,
  WorkspaceFile,
} from "./types";

export function getCatalog(): Promise<CatalogOut> {
  return apiFetch<CatalogOut>("/catalog");
}

export function getTopic(topicId: string): Promise<TopicOut> {
  return apiFetch<TopicOut>(`/topics/${encodeId(topicId)}`);
}

export function getProblem(problemId: string): Promise<ProblemOut> {
  return apiFetch<ProblemOut>(`/problems/${encodeId(problemId)}`);
}

export function getWorkspace(problemId: string): Promise<WorkspaceFile> {
  return apiFetch<WorkspaceFile>(`/workspace/${encodeId(problemId)}`);
}

export function putWorkspace(problemId: string, content: string): Promise<WorkspaceFile> {
  return apiFetch<WorkspaceFile>(`/workspace/${encodeId(problemId)}`, {
    method: "PUT",
    body: JSON.stringify({ content }),
  });
}

export function postAttempt(problemId: string, mode: RunMode): Promise<AttemptCreated> {
  return apiFetch<AttemptCreated>("/attempts", {
    method: "POST",
    body: JSON.stringify({ problem_id: problemId, mode }),
  });
}

export function getProgress(): Promise<ProgressOut> {
  return apiFetch<ProgressOut>("/progress");
}
