// Mock interview state and its mutations. Every write refreshes the open-mock status, so the page
// moves from the plan to the scoring form, to the debrief, without a reload.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  abandonMock,
  getDebrief,
  getMockStatus,
  revealMock,
  scoreMock,
  startMock,
} from "../api/mocks";

export const MOCK_STATUS_KEY = ["mock", "current"] as const;

export function useMockStatus() {
  return useQuery({ queryKey: MOCK_STATUS_KEY, queryFn: getMockStatus, retry: false });
}

export function useMockActions() {
  const client = useQueryClient();
  const refresh = () => void client.invalidateQueries({ queryKey: ["mock"] });
  const start = useMutation({
    mutationFn: (caseId: string | undefined) => startMock(caseId),
    retry: false,
    onSuccess: refresh,
  });
  const reveal = useMutation({
    mutationFn: (question: string) => revealMock(question),
    retry: false,
  });
  const score = useMutation({
    mutationFn: (scores: Record<string, number>) => scoreMock(scores),
    retry: false,
    onSuccess: refresh,
  });
  const abandon = useMutation({
    mutationFn: () => abandonMock(),
    retry: false,
    onSuccess: refresh,
  });
  return { start, reveal, score, abandon };
}

export function useDebrief(sessionId: string | null) {
  return useQuery({
    queryKey: ["mock", "debrief", sessionId],
    queryFn: () => getDebrief(sessionId ?? ""),
    enabled: sessionId !== null,
    retry: false,
  });
}
