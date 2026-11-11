-- Correct, different shape: orders are summed per customer in a subquery, then joined to the current customers.
SELECT
    c.customer_id,
    c.customer_name,
    COALESCE(o.order_count, 0) AS order_count,
    COALESCE(o.total_amount, 0.0) AS total_amount
FROM customers AS c
LEFT JOIN (
    SELECT
        customer_id,
        COUNT(order_id) AS order_count,
        SUM(amount) AS total_amount
    FROM orders
    GROUP BY customer_id
) AS o
    ON o.customer_id = c.customer_id
WHERE c.is_current
ORDER BY total_amount DESC, c.customer_id ASC;
