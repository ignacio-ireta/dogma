// The calendar's wording, in the learner's terms (ADR-0019). Never a pace judgement,
// or a count of days. Dates follow `dogma warp`: "Mon 09 Nov", with the year where it helps.
import type { CalendarPointOut } from "../../api/calendar";

export function dayLabel(iso: string, withYear = false): string {
  const date = new Date(`${iso}T00:00:00Z`);
  const day = new Intl.DateTimeFormat("en-GB", {
    weekday: "short",
    day: "2-digit",
    month: "short",
    timeZone: "UTC",
  }).format(date);
  return withYear ? `${day} ${date.getUTCFullYear()}` : day;
}

export function thisWeekLine(current: number, total: number, end: string): string {
  return `This week is week ${current} of ${total} · the course ends ${dayLabel(end, true)}`;
}

export function furthestLine(anchorWeek: number, anchorProblemId: string | null): string {
  if (anchorProblemId === null) return "You haven't run an exercise yet.";
  return `Your furthest exercise is in week ${anchorWeek}.`;
}

export function offsetLine(offsetWeeks: number): string | null {
  if (offsetWeeks === 0) return null;
  const count = Math.abs(offsetWeeks);
  const unit = count === 1 ? "week" : "weeks";
  const direction = offsetWeeks > 0 ? "later" : "earlier";
  return `Dates moved ${count} ${unit} ${direction} by time-warp.`;
}

export interface ChangeLine {
  readonly label: string;
  readonly before: string;
  readonly after: string;
}

/** The before → after lines a preview shows. */
export function changeLines(before: CalendarPointOut, after: CalendarPointOut): ChangeLine[] {
  return [
    {
      label: "This week",
      before: `week ${before.current_week}`,
      after: `week ${after.current_week}`,
    },
    {
      label: "Weekly mocks from",
      before: dayLabel(before.mocks_from),
      after: dayLabel(after.mocks_from),
    },
    {
      label: "Interview week",
      before: dayLabel(before.interview_week),
      after: dayLabel(after.interview_week),
    },
    {
      label: "Course ends",
      before: dayLabel(before.end, true),
      after: dayLabel(after.end, true),
    },
  ];
}
