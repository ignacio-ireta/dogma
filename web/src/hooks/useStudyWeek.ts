import { useQuery } from "@tanstack/react-query";
import { getProgress } from "../api/endpoints";
import type { ProgressOut } from "../api/types";

export interface StudyWeek {
  readonly week: number;
  readonly theme: string | null;
}

/** The study week the header shows. Read-only: progress is owned by the API. */
export function useStudyWeek(): { readonly week: StudyWeek | null; readonly isError: boolean } {
  const query = useQuery({ queryKey: ["progress"], queryFn: getProgress, retry: false });
  return {
    week: query.data ? studyWeek(query.data) : null,
    isError: query.isError,
  };
}

export function studyWeek(progress: ProgressOut): StudyWeek {
  const current = progress.summary.current_week;
  const row = progress.weeks.find((item) => item.week === current);
  return { week: current, theme: row?.theme ?? null };
}
