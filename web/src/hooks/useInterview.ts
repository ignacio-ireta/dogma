// Interview questions, the topic-answer rubric and the answer mutation. Saving an answer refreshes
// the topic's mastery and the progress readiness, which both change when a topic is promoted.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getInterview, getRubric, postAnswer } from "../api/interview";
import type { AnswerRequest } from "../api/types";

export function useInterview(topicId: string) {
  return useQuery({
    queryKey: ["interview", topicId],
    queryFn: () => getInterview(topicId),
    enabled: topicId !== "",
    retry: false,
  });
}

export function useRubric(rubricId: "topic-answer" | "mock") {
  return useQuery({
    queryKey: ["rubric", rubricId],
    queryFn: () => getRubric(rubricId),
    staleTime: Number.POSITIVE_INFINITY,
    retry: false,
  });
}

export function useAnswerMutation(topicId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: AnswerRequest) => postAnswer(topicId, body),
    retry: false,
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["interview", topicId] });
      void client.invalidateQueries({ queryKey: ["progress"] });
    },
  });
}
