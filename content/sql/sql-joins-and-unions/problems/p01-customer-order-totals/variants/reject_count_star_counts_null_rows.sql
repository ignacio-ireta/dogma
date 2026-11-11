-- Wrong on purpose: COUNT(*) counts the unmatched left-join row, so no-order customers get 1.
SELECT
    c.customer_id,
    c.customer_name,
    COUNT(*) AS order_count,
    COALESCE(SUM(o.amount), 0.0) AS total_amount
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id
WHERE c.is_current
GROUP BY c.customer_id, c.customer_name
ORDER BY total_amount DESC, c.customer_id ASC;
