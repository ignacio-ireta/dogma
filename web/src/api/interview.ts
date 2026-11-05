// Interview answer and rubric calls (ARCHITECTURE §4.4). Components reach these through hooks.
import { apiFetch, encodeId } from "./client";
import type { AnswerOut, AnswerRequest, InterviewOut, RubricOut } from "./types";

export function getInterview(topicId: string): Promise<InterviewOut> {
  return apiFetch<InterviewOut>(`/topics/${encodeId(topicId)}/interview`);
}

export function postAnswer(topicId: string, body: AnswerRequest): Promise<AnswerOut> {
  return apiFetch<AnswerOut>(`/topics/${encodeId(topicId)}/answers`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getRubric(rubricId: "topic-answer" | "mock"): Promise<RubricOut> {
  return apiFetch<RubricOut>(`/rubrics/${rubricId}`);
}
