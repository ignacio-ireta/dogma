// The open timed re-solve, if any. Shared by Today (the start action), the Problem page (budget and
// hint warning) and the mutations that start, abandon or close a window.
import { useQuery } from "@tanstack/react-query";
import { getCurrentResolve } from "../api/resolves";

export const CURRENT_RESOLVE_KEY = ["resolve", "current"] as const;

export function useCurrentResolve() {
  return useQuery({
    queryKey: CURRENT_RESOLVE_KEY,
    queryFn: getCurrentResolve,
    retry: false,
    refetchOnWindowFocus: false,
  });
}
