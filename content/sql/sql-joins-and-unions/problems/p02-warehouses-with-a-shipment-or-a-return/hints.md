## Hint 1 — concept

A union stacks the rows of two queries that return the same columns. The question to answer first is
whether duplicates should survive, because that decides between two set operators.

## Hint 2 — formula / approach

Select the code from `shipments` and from `returns`, each branch with `WHERE warehouse_code IS NOT NULL`.
Join the two branches with `UNION`, which removes duplicates, not with `UNION ALL`, which keeps them.
A `JOIN` would keep only codes that are in both tables, which is the wrong question.

## Hint 3 — code shape

```sql
SELECT warehouse_code
FROM ...
WHERE ...
UNION              -- or UNION ALL: which one removes duplicates?
SELECT warehouse_code
FROM ...
WHERE ...
ORDER BY ...;      -- after the union, so it sorts the combined result
```
