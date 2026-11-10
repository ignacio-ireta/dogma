# Channel conversion and device revenue

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that summarises each marketing
channel's sessions. For every channel, return how many sessions it had, how many converted, the
conversion rate, and the revenue from mobile sessions and from desktop sessions as two separate
columns. Every number comes from one pass over the table, with no join and no second query.

## Contract

- Channel values are grouped exactly as stored; casing and surrounding whitespace are not normalized.
- Input: one table, `sessions`.
  - `session_id`: BIGINT, unique.
  - `channel`: VARCHAR, never NULL. Lowercase names such as `ads`.
  - `device`: VARCHAR, never NULL. Either `'mobile'` or `'desktop'`.
  - `converted`: BOOLEAN, never NULL. TRUE when the session converted.
  - `revenue`: DOUBLE, may be NULL. The revenue recorded on the session. NULL means no revenue is
    recorded, and it adds nothing to a sum.
- Returns: one row per channel that has at least one session, with exactly these columns in
  this order:
  - `channel`: the channel name.
  - `sessions`: the number of sessions in the channel, on every device, converted or not.
  - `converted_sessions`: the number of those sessions with `converted = TRUE`. `0` when none.
  - `conversion_rate`: `converted_sessions / sessions` as a fraction (0.25 means 25%), rounded to
    4 decimal places with `ROUND(x, 4)`. `0.0` when no session converted. Never NULL.
  - `mobile_revenue`: the sum of `revenue` over the channel's mobile sessions, converted or not.
    `0.0` when the channel has no mobile session with a recorded revenue. Never NULL.
  - `desktop_revenue`: the same sum over the channel's desktop sessions.
- Row order: `channel` ascending.
- If `sessions` has no rows, return zero rows with the same six columns.
- Must read from `sessions`. Do not write the answer as literal rows.

## Examples

Input, six sessions across two channels:

```text
session_id  channel  device   converted  revenue
1           ads      mobile   TRUE       40.0
2           ads      desktop  FALSE      NULL
3           ads      mobile   FALSE      NULL
4           ads      desktop  TRUE       NULL
5           ads      mobile   TRUE       25.0
6           email    desktop  FALSE      NULL
```

Output:

```text
channel  sessions  converted_sessions  conversion_rate  mobile_revenue  desktop_revenue
ads      5         3                   0.6              65.0            0.0
email    1         0                   0.0              0.0             0.0
```

Session 4 converted but has no recorded revenue, so it counts as converted and adds nothing to
either revenue column. Email has no mobile session at all, so its mobile revenue is `0.0`, not NULL.

On `fixtures/sessions.csv`, the answer is:

```text
channel   sessions  converted_sessions  conversion_rate  mobile_revenue  desktop_revenue
ads       8         3                   0.375            65.0            0.0
email     6         2                   0.3333           30.0            15.0
organic   5         0                   0.0              0.0             0.0
podcast   2         1                   0.5              0.0             12.5
referral  3         1                   0.3333           80.0            0.0
```

A generated event log supplies dirty channel labels and lognormal amounts. Exact stored labels
form separate groups, and each group's revenue can have a long right tail.

## Constraints

- Allowed: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `CASE`, `FILTER (WHERE ...)`,
  `CAST`, `COALESCE`, `NULLIF`, `ROUND`, and DuckDB's built-in aggregates and arithmetic.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `sessions`
tables, so a query tuned to the fixture fails. The hidden tests cover these families: channels
with no conversions, a channel that has sessions on one device only, revenue that is NULL on every
session of a device, a converted session with no revenue, a rate that must be a fraction (not a
whole number or a percentage) rounded to 4 places, revenue counted on non-converted sessions, an
empty table, the row order, dirty channel keys kept as stored, lognormal revenue, and generated event-log
rows checked against an independent pandas computation.
