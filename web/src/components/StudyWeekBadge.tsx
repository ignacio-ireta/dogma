import { useStudyWeek } from "../hooks/useStudyWeek";

/** The study week, read-only, from the API. Hidden when the API cannot say (no progress yet). */
export function StudyWeekBadge() {
  const { week } = useStudyWeek();
  if (week === null) {
    return null;
  }
  const theme = week.theme === null ? "" : ` · ${week.theme}`;
  return (
    <span className="study-week">
      Week {week.week}
      {theme}
    </span>
  );
}
