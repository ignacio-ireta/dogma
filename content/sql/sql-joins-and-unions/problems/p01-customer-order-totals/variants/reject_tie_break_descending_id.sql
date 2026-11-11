-- Wrong on purpose: equal totals are ordered by customer_id descending instead of ascending.
SELECT
    c.customer_id,
    c.customer_name,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.amount), 0.0) AS total_amount
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
WHERE c.is_current
GROUP BY c.customer_id, c.customer_name
ORDER BY total_amount DESC, c.customer_id DESC;
