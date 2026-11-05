# Total spend by user

Write a DuckDB query over `orders(user_id INTEGER, amount DOUBLE)` that returns one row per
`user_id` with the total `amount`, named `total`.

## Contract

- Input: table `orders`. May be empty.
- Returns: columns `user_id` and `total`, one row per user. Row order does not matter.
- Raises: nothing for valid input.
- Must not mutate the table.

## Examples

```sql
-- orders: (1, 2.5), (1, 1.5), (2, 4.0)
-- result: (1, 4.0), (2, 4.0)
```

```sql
-- orders: empty
-- result: no rows
```

## Constraints

- Allowed: SQL that DuckDB accepts. Not allowed: subqueries that read other tables.

## Done looks like

All public tests pass, then the hidden tests pass. They check the grouped totals on a fixture,
an empty table, users with a single order, and a NULL amount.
