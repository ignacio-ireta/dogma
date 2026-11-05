## Hint 1 — concept

A window function ranks each row against the other rows in its partition, without collapsing them the
way GROUP BY does. Every product stays in the output. The three rank functions differ only in how they
treat ties, so the question to answer first is: when two revenues are equal, should the rows share a
rank, or should one come first?

## Hint 2 — formula / approach

Write one `OVER (PARTITION BY category ORDER BY revenue DESC NULLS LAST ...)` per function. `ROW_NUMBER`
needs a unique order, so add `product_id` as a second sort key. `RANK` and `DENSE_RANK` must keep ties
as peers, so give them only the revenue key. Finish with `ORDER BY category, row_number`.

## Hint 3 — code shape

```sql
SELECT
    category, product_id, revenue,
    ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ..., product_id) AS row_number,
    RANK()       OVER (PARTITION BY ... ORDER BY ...)             AS rank,
    DENSE_RANK() OVER (PARTITION BY ... ORDER BY ...)             AS dense_rank
FROM products
ORDER BY ...;
```

Fill the three `...` with the partition, the revenue key and the tie-break. Check the NULL position
explicitly with `NULLS LAST`.
