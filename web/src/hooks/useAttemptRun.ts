// Run and Submit: save the buffer, then grade the file on disk (the API's recording rule applies).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { LocalError } from "../api/client";
import { postAttempt } from "../api/endpoints";
import type { RunMode } from "../api/types";

export const UNSAVED_MESSAGE =
  "The file is not saved, so it was not graded. Fix the save problem, then run again.";

export function useAttemptRun(problemId: string, saveNow: () => Promise<boolean>) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (mode: RunMode) => {
      const saved = await saveNow();
      if (!saved) {
        throw new LocalError(UNSAVED_MESSAGE);
      }
      return postAttempt(problemId, mode);
    },
    // A submit may close the open timed re-solve, so the window is read again after every grade.
    onSettled: () => client.invalidateQueries({ queryKey: ["resolve"] }),
  });
}
