// Mock interview calls (ARCHITECTURE §4.4). The open mock is read with getMockStatus.
import { apiFetch } from "./client";
import type {
  AbandonedOut,
  MockDebriefOut,
  MockPlanOut,
  MockReportOut,
  MockStatusOut,
  RevealOut,
} from "./types";

export function getMockStatus(): Promise<MockStatusOut> {
  return apiFetch<MockStatusOut>("/mocks/current");
}

export function startMock(caseId?: string): Promise<MockPlanOut> {
  return apiFetch<MockPlanOut>("/mocks", {
    method: "POST",
    body: JSON.stringify(caseId ? { case_id: caseId } : {}),
  });
}

export function revealMock(question: string): Promise<RevealOut> {
  return apiFetch<RevealOut>("/mocks/current/reveal", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export function scoreMock(scores: Record<string, number>): Promise<MockReportOut> {
  return apiFetch<MockReportOut>("/mocks/current/score", {
    method: "POST",
    body: JSON.stringify({ scores }),
  });
}

export function abandonMock(): Promise<AbandonedOut> {
  return apiFetch<AbandonedOut>("/mocks/current/abandon", { method: "POST" });
}

export function getDebrief(sessionId: string): Promise<MockDebriefOut> {
  return apiFetch<MockDebriefOut>(`/mocks/${encodeURIComponent(sessionId)}/debrief`);
}
