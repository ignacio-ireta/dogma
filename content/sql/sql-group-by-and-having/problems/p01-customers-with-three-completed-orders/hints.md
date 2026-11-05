## Hint 1 — concept

Before grouping, decide which rows count at all. After grouping, decide which groups survive.
Those are different stages of the query, and each condition belongs in one of them.

## Hint 2 — formula / approach

Filter rows with `WHERE status = 'completed'`. Group by `customer_id`. Count with `COUNT(*)`,
which counts a NULL-amount order. Sum with `SUM(amount)`, which skips NULLs. Keep groups with
`HAVING COUNT(*) >= 3`. Sort by the total with `NULLS LAST`, then by `customer_id`.

## Hint 3 — code shape

```sql
SELECT customer_id, COUNT(*) AS ..., SUM(...) AS ...
FROM orders
WHERE ...            -- rows: which orders count
GROUP BY ...         -- one row per customer
HAVING ...           -- groups: which customers survive
ORDER BY ...         -- total descending with NULLs last, then customer
;
```
