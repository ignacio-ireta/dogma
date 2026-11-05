# Interview notes: SQL date handling

## Q1. Give me monthly active users for every month in a range, including months with no activity.

**60-second answer.** I start from a calendar: one row per month of the range, generated with
`generate_series` and cast to DATE, so the calendar is the left side of the join. Then I LEFT JOIN the events
with a half-open test, `event_ts >= month AND event_ts < month + 1 month`, so each event lands in exactly one
month and a boundary instant is never counted twice. I count `COUNT(DISTINCT user_id)`, so a user with several
events in a month counts once, and an empty month gets 0. I check the first and last months of the range, a
month with no events in the middle, an event at midnight on the 1st, and an event at 23:59:59 on the last day.
The cost depends on the join plan and the distinct aggregation; inspect the plan on large tables.

**Follow-ups.**
- *What changes if the events carry a time zone?* Convert each timestamp to one reporting zone before
  truncating, because the month boundary depends on the zone.
- *Why not a window function?* A window function does not create rows for months with no events. The calendar
  does.

## Q2. For each user, how many days passed between the first and second event?

**60-second answer.** I number each user's events by time with `ROW_NUMBER` over the user, then join position
1 to position 2 on `user_id`. Users with one event drop out through the inner join. The distance is a DATE
difference, `CAST(second AS DATE) - CAST(first AS DATE)`, so it counts calendar days. Subtracting two timestamps
would give an interval, and reading its day field counts elapsed days, so a span of two minutes across midnight
would come out as 0. Ties need a rule. Two events with the same timestamp are two events, so their distance is 0.

**Follow-ups.**
- *What if a user has three events?* The second event is the answer, not the last one. MAX minus MIN is wrong.
- *Could you avoid window functions?* Yes: take the first timestamp with MIN, then compare the count of events
  at that timestamp with 2. If there are at least two, the second event shares it. Otherwise take the smallest
  timestamp after it.

## Common mistakes
- Using `BETWEEN` on timestamps, which counts the next period's first instant in the previous period.
- Building the result from the events alone, so empty periods disappear.
- Using `range()` where `generate_series()` is meant, which loses the last period.
- Subtracting timestamps and reading the day field, which counts elapsed days instead of calendar days.
- Returning a TIMESTAMP where the contract asks for a DATE, because `date_trunc` returns a TIMESTAMP for a DATE.
