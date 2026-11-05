## Hint 1 — concept

Think in the order the engine evaluates a `SELECT`: rows are filtered by `WHERE` first, then
sorted, then cut down by `LIMIT`. Anything that should decide membership in the answer belongs
in `WHERE`. Anything that decides order or the cut belongs after it.

## Hint 2 — formula / approach

Three conditions go in `WHERE`: exact status, the 2024 date range written as a half-open range
(`>= 2024-01-01` and `< 2025-01-01`), and `amount IS NOT NULL`. Then `ORDER BY amount DESC,
order_id ASC` and `LIMIT 5`. The second sort key makes the answer deterministic when amounts tie.

## Hint 3 — code shape

```sql
SELECT order_id, customer_id, order_date, amount
FROM orders
WHERE ...            -- three conditions joined with AND
ORDER BY ...         -- two keys, each with its direction
LIMIT 5;
```
