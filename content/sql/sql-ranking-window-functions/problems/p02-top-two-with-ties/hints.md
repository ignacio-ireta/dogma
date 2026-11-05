## Hint 1 — concept

A top-N query asks for the first N rows of each group. The question that matters here is what "first N"
means when values tie. With a dense rank, the second rank is the second distinct revenue, so every product
that shares it belongs in the answer, and the result can be longer than N.

## Hint 2 — formula / approach

Give each product its dense rank inside its category, ordered by revenue descending, with the NULL revenue
last. Keep the rows whose dense rank is at most 2. The window function has to be computed first, because a
window function cannot be filtered in the same `WHERE` that defines it.

## Hint 3 — code shape

```sql
SELECT category, product_id, revenue, dense_rank
FROM (
    SELECT ..., DENSE_RANK() OVER (PARTITION BY ... ORDER BY ...) AS dense_rank
    FROM products
) AS ranked
WHERE ...
ORDER BY ...;
```

Name the inner query, filter on its alias in the outer one, and finish with the three-part ORDER BY.
