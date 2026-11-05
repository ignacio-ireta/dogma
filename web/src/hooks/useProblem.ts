import { useQuery } from "@tanstack/react-query";
import { getProblem } from "../api/endpoints";

export function useProblem(problemId: string) {
  return useQuery({
    queryKey: ["problem", problemId],
    queryFn: () => getProblem(problemId),
    enabled: problemId !== "",
  });
}
