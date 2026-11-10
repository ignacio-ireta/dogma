# First and last event per user

Return one row for each user who has at least one event in the `events` table. Each row gives the time and the
type of that user's first event, and the time and the type of the last event.

## Contract

Table `events` has these columns:

| Column | DuckDB type | Rules |
|---|---|---|
| user_id | BIGINT | Non-null grouping key |
| event_id | BIGINT | Non-null, globally unique |
| event_time | TIMESTAMP | Non-null; no time-zone conversion |
| event_type | VARCHAR | Non-null |

- **First event** of a user: the smallest `event_time`. When several events share it, the one with the
  **smallest** `event_id`.
- **Last event** of a user: the largest `event_time`. When several events share it, the one with the
  **largest** `event_id`.
- Output columns, in this order: `user_id BIGINT`, `first_event_time TIMESTAMP`, `first_event_type VARCHAR`,
  `last_event_time TIMESTAMP`, `last_event_type VARCHAR`.
- Exactly one row per user. The times and types of a row come from the same event.
- Final output order is `user_id ASC`.
- A user with a single event has the same event as first and last.
- `event_type` is returned exactly as stored: its case and any surrounding spaces are part of the
  value. Labels are not trimmed or case-folded.
- An empty table returns an empty result with these column names and types.
- Do not change the input table.

## Examples

```text
user_id event_id event_time        event_type
1       1        2026-01-01 10:00  login
1       2        2026-01-01 10:00  view
1       3        2026-01-02 09:00  logout
1       4        2026-01-02 09:00  purchase
2       5        2026-01-03 08:00  view
=> (1, 2026-01-01 10:00, login, 2026-01-02 09:00, purchase),
   (2, 2026-01-03 08:00, view, 2026-01-03 08:00, view)
```

```text
user_id event_id event_time        event_type
7       12       2026-02-01 00:00  signup
7       10       2026-02-01 00:00  visit
7       11       2026-01-15 12:30  invite
=> (7, 2026-01-15 12:30, invite, 2026-02-01 00:00, signup)
```

In the second example, event 11 is first because its time is earliest, even though its id is in the middle. Events
10 and 12 tie for last at 2026-02-01 00:00, and the largest id, 12, wins.

```text
user_id event_id event_time        event_type
4       21       2026-03-05 14:00  " Login"
4       22       2026-03-05 16:00  "LOGIN "
=> (4, 2026-03-05 14:00, " Login", 2026-03-05 16:00, "LOGIN ")
```

The labels keep their case and spaces, because the contract returns the stored value.

## Constraints

- Use a read-only DuckDB SQL query. CTEs, subqueries, window functions, `arg_min`, `arg_max` and `struct_pack`
  are all allowed.
- The query is graded by its result, not by its text.

## Done looks like

The result matches an independent pandas oracle on seeded and fixture events, and on a skewed log (a few heavy users,
second-resolution times) whose labels carry case and spacing noise. Hidden cases cover ties at the first time and at
the last time, users with one event, event ids out of time order, a user whose events all share one time, large
identifiers, empty input, the exact column types and the order of the output rows. Labels come back exactly as stored.
Every returned time and type belongs to the same event.
