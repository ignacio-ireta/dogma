import { useQuery } from "@tanstack/react-query";
import { getTopic } from "../api/endpoints";

export function useTopic(topicId: string) {
  return useQuery({
    queryKey: ["topic", topicId],
    queryFn: () => getTopic(topicId),
    enabled: topicId !== "",
  });
}
