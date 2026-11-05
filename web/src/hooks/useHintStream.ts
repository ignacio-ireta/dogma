// A hint request as a stream: the reviewed text at once, the nudge as it arrives, then the final
// text. A new request aborts the previous one; leaving the page aborts the open one.
import { useCallback, useEffect, useReducer, useRef } from "react";
import { type HintEvent, hintReducer, initialHintState, parseHintEvent } from "../api/hint-stream";
import { postEventStream } from "../api/sse";

export function useHintStream(problemId: string) {
  const [state, dispatch] = useReducer(hintReducer, initialHintState);
  const active = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      active.current?.abort();
    };
  }, []);

  const request = useCallback(
    async (level: number, confirmWalkthrough: boolean) => {
      active.current?.abort();
      const controller = new AbortController();
      active.current = controller;
      dispatch({ type: "start", level });
      try {
        await postEventStream(
          "/tutor/hint/stream",
          { problem_id: problemId, level, confirm_walkthrough: confirmWalkthrough },
          {
            signal: controller.signal,
            onEvent: (raw) => {
              if (controller.signal.aborted) return;
              const event: HintEvent | null = parseHintEvent(raw);
              if (event !== null) {
                dispatch({ type: "event", event });
              }
            },
          },
        );
        if (!controller.signal.aborted) {
          dispatch({ type: "ended" });
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          dispatch({ type: "failed", error });
        }
      }
    },
    [problemId],
  );

  return { state, request };
}
