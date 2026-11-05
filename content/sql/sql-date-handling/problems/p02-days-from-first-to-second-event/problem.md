# Days from a user's first event to their second

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns one row for each user who has at
least two events: the number of calendar days between the day of the user's first event and the day of
their second event. A user with one event is left out.

## Contract

- Input: one table `events`.
  - `user_id` (BIGINT): the user. Never NULL.
  - `event_ts` (TIMESTAMP): when the event happened. Naive, with no time zone. Never NULL.
  - One row per event. Two events of one user can share a timestamp; they are still two events.
- A user's first event is the earliest `event_ts` of that user. Their second event is the next one in time
  order. If the first two timestamps are equal, the second event has that same timestamp.
- `days_to_second`: BIGINT, the number of calendar days from the first event's date to the second event's
  date. Events at `2024-03-31 23:59:00` and `2024-04-01 00:01:00` are 1 day apart, although only 2 minutes
  separate them. The value is never negative.
- Returns: one row per user with at least two events, with exactly these columns in this order:
  `user_id`, `days_to_second`.
- Row order: `user_id` ascending.
- If no user has two events, return zero rows with the same two columns.
- Must read from `events`. Do not write the answer as literal rows.

## Examples

On `fixtures/events.csv`, the answer is:

```text
user_id  days_to_second
201      3
202      1
204      0
205      1
206      59
```

User 203 has one event and is not returned. User 204 has two events at the same timestamp, so the second
event is on the same day as the first. User 206 spans the leap day: from 2024-01-01 to 2024-02-29 is 59 days.

A small case: user 9 has events at 2024-12-31 23:00:00, 2025-01-01 01:00:00 and 2025-01-10 08:00:00. User
10 has one event.

```text
user_id  event_ts
9        2025-01-01 01:00:00
9        2024-12-31 23:00:00
9        2025-01-10 08:00:00
10       2025-01-02 12:00:00

user_id  days_to_second
9        1
```

## Constraints

- Allowed: `SELECT`, `FROM`, `JOIN`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `WITH`, `ROW_NUMBER() OVER`,
  `LEAD`, `QUALIFY`, DuckDB's date functions and built-in aggregates.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `events` tables. The
families they cover: users with one event, a tie at the first timestamp, users with three or more events
(the second event, not the last), a span of two minutes across midnight, a span across the leap day, events
given out of time order, large non-sequential user ids, a result with no user who has two events, an empty
events table, the direction of the difference, and random tables checked against an independent pandas
computation.
