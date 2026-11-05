# Day-over-day change per account

Table `account_daily_usage` records billable API calls per account per day. A row exists only
for a day on which the account made at least one call. For every row, return the account's
previous recorded day and the change in calls since that day. The first recorded day for each
account has no predecessor, so its previous-day fields and change are NULL.

## Contract

Table `account_daily_usage` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| account_id | BIGINT | Non-null grouping key |
| usage_date | DATE | Non-null. Unique within an account |
| api_calls | BIGINT | Non-null, at least 1 |

Return these columns, in this order and with these types:

| Column | DuckDB type | Meaning |
|---|---|---|
| account_id | BIGINT | Same as input |
| usage_date | DATE | Same as input |
| api_calls | BIGINT | Same as input |
| previous_usage_date | DATE | The account's previous recorded day, or NULL |
| previous_api_calls | BIGINT | Calls on that previous day, or NULL |
| calls_change | BIGINT | `api_calls - previous_api_calls`, or NULL |

- "Previous" means the previous recorded row for the same account. Do not fill in missing
  calendar days; a gap of two days is still a one-row step.
- A missing predecessor is NULL, never 0.
- The change is current minus previous. A decrease is negative.
- Final output order is `account_id ASC, usage_date ASC`.
- An empty table returns no rows, with the stated column names and types.
- Do not change the input table.

## Examples

Input:

```text
account_id  usage_date  api_calls
1           2026-03-01  100
1           2026-03-02  150
1           2026-03-05  120
2           2026-03-02  40
```

Result rows, with the six columns of the contract in order:

```text
account_id  usage_date  api_calls  previous_usage_date  previous_api_calls  calls_change
1           2026-03-01  100        NULL                 NULL                NULL
1           2026-03-02  150        2026-03-01           100                 50
1           2026-03-05  120        2026-03-02           150                 -30
2           2026-03-02  40         NULL                 NULL                NULL
```

The 2026-03-05 row skips two calendar days, but its predecessor is the 2026-03-02 row.

A second example, with an account that has one recorded day and a drop:

```text
account_id  usage_date  api_calls   =>   previous_usage_date  previous_api_calls  calls_change
9           2026-07-14  500              NULL                 NULL                NULL
9           2026-07-15  125              2026-07-14           500                 -375
```

## Constraints

Use a read-only DuckDB SQL query. Window functions, CTEs and subqueries are allowed. The
query is graded by its results, not by a particular syntax. Do not hard-code dates or account
identifiers.

## Done looks like

The result matches a pandas `groupby(...).shift(1)` oracle on seeded random usage with gaps
and shuffled input order. Hidden cases cover the NULL first row, skipped calendar days,
decreases, output ordering that does not follow input order, a single-day account, large
BIGINT values, the empty table and the exact column types.
