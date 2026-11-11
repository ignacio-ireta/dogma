-- Wrong on purpose: a NULL amount of the customer is counted as 0 in the average.
--
-- The scalar subquery in the SELECT list is correlated: it refers to o.customer_id, which comes
-- from the outer row, so it gives each order its own customer's average. AVG skips NULL amounts.
-- The outer query keeps the rows whose amount is above that average. The filter is written on the
-- derived table `scored`, because a column alias created in the SELECT list cannot be used in the
-- WHERE clause of the same query in standard SQL.
SELECT
    order_id,
    customer_id,
    amount,
    customer_average
FROM (
    SELECT
        o.order_id,
        o.customer_id,
        o.amount,
        (
            SELECT AVG(COALESCE(i.amount, 0))
            FROM orders AS i
            WHERE i.customer_id = o.customer_id   -- the correlation: this customer only
              AND i.status = 'completed'
        ) AS customer_average
    FROM orders AS o
    WHERE o.status = 'completed'
      AND o.customer_id IS NOT NULL
      AND o.amount IS NOT NULL
) AS scored
WHERE amount > customer_average                    -- strict: an order equal to its average is out
ORDER BY order_id ASC;
