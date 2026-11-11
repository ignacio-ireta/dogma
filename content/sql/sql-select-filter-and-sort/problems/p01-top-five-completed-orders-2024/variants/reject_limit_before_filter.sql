-- Wrong on purpose: the top five orders overall are taken first and only then filtered, so fewer than five can come back.
SELECT order_id, customer_id, order_date, amount
FROM (
    SELECT order_id, customer_id, order_date, amount, status
    FROM orders
    WHERE amount IS NOT NULL
    ORDER BY amount DESC, order_id ASC
    LIMIT 5
) AS top
WHERE status = 'completed'
  AND order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2025-01-01'
ORDER BY amount DESC, order_id ASC;
