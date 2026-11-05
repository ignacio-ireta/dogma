## Hint 1 — concept

Decide the grain before writing any join. The answer has one row per current customer, so the
customer side decides which rows exist. Orders can only add to a customer, never create one, so an
order whose customer is missing from the customer table should not appear anywhere.

## Hint 2 — formula / approach

Filter `customers` to the current version before the join, or the older versions will repeat every
order. Then `LEFT JOIN` the orders onto that customer side. Count the order key (`COUNT(o.order_id)`),
not the rows: a customer with no orders still produces one joined row, with NULL order columns, and
that row must count as zero. Wrap the sum in `COALESCE(..., 0.0)`.

## Hint 3 — code shape

```sql
SELECT c.customer_id, c.customer_name, COUNT(...) AS order_count, COALESCE(SUM(...), 0.0) AS total_amount
FROM customers AS c
LEFT JOIN orders AS o ON ...     -- which orders belong to which customer
WHERE ...                         -- which customer rows count
GROUP BY ...                      -- one row per customer
ORDER BY ...                      -- total descending, then customer id
;
```
