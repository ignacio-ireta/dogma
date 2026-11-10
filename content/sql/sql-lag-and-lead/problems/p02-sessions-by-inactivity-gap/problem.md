# Sessions by inactivity gap

Table `screen_views` records one row per screen view in a mobile app. Split each visitor's views
into sessions: a new session starts at a visitor's first view, and at any view whose gap from
the visitor's previous view is greater than 30 minutes. Return every view with its session number
and a flag that marks where a session starts.

## Contract

Table `screen_views` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| visitor_id | BIGINT | Non-null |
| view_id | BIGINT | Non-null, globally unique |
| viewed_at | TIMESTAMP | Non-null. Several views of one visitor may share a timestamp |

Return these columns, in this order and with these types:

| Column | DuckDB type | Meaning |
|---|---|---|
| visitor_id | BIGINT | Same as input |
| view_id | BIGINT | Same as input |
| viewed_at | TIMESTAMP | Same as input |
| session_number | BIGINT | 1 for the visitor's first session, incremented at each session start |
| starts_session | BOOLEAN | TRUE only on the view that opens a session |

The types are checked exactly. `session_number` must be `BIGINT`, although DuckDB's `SUM` over
integers returns a wider integer type by default, so the result needs an explicit type.

- Order each visitor's views by `viewed_at ASC`, then by `view_id ASC`. That order defines
  "previous" and the running count. Equal timestamps are ordered by the lower `view_id` first.
- The gap is `viewed_at` minus the previous view's `viewed_at`. A gap of exactly 30 minutes
  does not start a session. A gap of 30 minutes and one second does.
- The first view of a visitor always starts session 1.
- Timestamps keep their fractions of a second, so the gap is exact to the microsecond. A gap of 30
  minutes and one millisecond does start a session.
- Session numbers restart at 1 for each visitor.
- Final output order is `visitor_id ASC, viewed_at ASC, view_id ASC`.
- Every input row appears exactly once. An empty table returns no rows, with the stated schema.
- Do not change the input table.

## Examples

Input rows for visitor 5, in time order:

```text
visitor_id  view_id  viewed_at
5           1        10:00
5           2        10:30    -- gap of exactly 30 minutes: same session
5           3        11:01    -- gap of 31 minutes: new session
```

Result rows:

```text
visitor_id  view_id  viewed_at  session_number  starts_session
5           1        10:00      1               TRUE
5           2        10:30      1               FALSE
5           3        11:01      2               TRUE
```

Ties at a boundary: visitor 6 has views 11 at 10:00, and views 13 and 12 both at 10:40.
View 12 is ordered first because its `view_id` is lower. It opens session 2, and view 13
continues it. The rule is about `view_id`, not about the order of the input rows.

Fractions of a second count. Visitor 7's second view comes 30 minutes and one millisecond after
the first, so it opens a new session:

```text
visitor_id  view_id  viewed_at     session_number  starts_session
7           21       07:00:00.200  1               TRUE
7           22       07:30:00.201  2               TRUE
```

## Constraints

Use a read-only DuckDB SQL query. Window functions, CTEs and subqueries are allowed. The query
is graded by its results, not by a particular syntax. Do not hard-code visitor identifiers.

## Done looks like

The result matches a pandas oracle (sort, group by visitor, shift, compare the gap, cumulative sum) on seeded random
views with exact 30-minute gaps, ties and visitors with one view, and on a Zipf-skewed visitor log with microsecond
clocks and gaps at the boundary (a synthetic generator). Hidden cases cover the boundary on both sides, a millisecond
past the boundary, the tie order that decides which row opens a session, session numbers restarting per visitor, the
BIGINT type of the session count, the output order, the empty table and the input table being unchanged.
