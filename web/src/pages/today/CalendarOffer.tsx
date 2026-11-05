// One neutral line when the calendar has run two or more weeks past the furthest exercise: a real
// break, not a normal Monday (ADR-0019). It links to the calendar on the Progress page.
import { Link } from "react-router";
import { Notice } from "../../components/Notice";
import { furthestLine } from "../progress/calendarText";
import { useCalendar } from "../progress/useCalendar";

/** Weeks past the furthest exercise before the offer shows. One week is ordinary. */
export const OFFER_GAP_WEEKS = 2;

export function CalendarOffer() {
  const { data } = useCalendar();
  if (!data || data.current_week < data.anchor_week + OFFER_GAP_WEEKS) return null;
  return (
    <Notice tone="info">
      {furthestLine(data.anchor_week, data.anchor_problem_id)}{" "}
      <Link to="/progress#calendar">
        Move the calendar so this week is week {data.anchor_week}?
      </Link>
    </Notice>
  );
}
