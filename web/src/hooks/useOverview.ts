import { useQuery } from "@tanstack/react-query";
import { getCoverage, getProgress, type Tier } from "../api/overview";

export function useCoverage(tier?: Tier, track?: string) {
  return useQuery({
    queryKey: ["coverage", tier ?? null, track ?? null],
    queryFn: () => getCoverage(tier, track),
  });
}

export function useOverviewProgress(week?: number) {
  return useQuery({
    queryKey: week === undefined ? ["progress"] : ["progress", week],
    queryFn: () => getProgress(week),
  });
}

/** Keep unknown server labels visible instead of assigning an invented state. */
export function recordedLabel(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("-", " ");
}
