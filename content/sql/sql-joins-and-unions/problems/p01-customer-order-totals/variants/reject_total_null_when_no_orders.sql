-- Wrong on purpose: a customer with no orders gets a NULL total instead of 0.0.
SELECT
    c.customer_id,
    c.customer_name,
    COUNT(o.order_id) AS order_count,
    SUM(o.amount) AS total_amount
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
WHERE c.is_current
GROUP BY c.customer_id, c.customer_name
ORDER BY total_amount DESC, c.customer_id ASC;
