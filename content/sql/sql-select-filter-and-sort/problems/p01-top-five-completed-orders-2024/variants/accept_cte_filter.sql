WITH qualifying AS (
    SELECT order_id, customer_id, order_date, amount
    FROM orders
    WHERE status = 'completed'
      AND order_date >= DATE '2024-01-01'
      AND order_date <  DATE '2025-01-01'
      AND amount IS NOT NULL
)
SELECT order_id, customer_id, order_date, amount
FROM qualifying
ORDER BY amount DESC, order_id ASC
LIMIT 5;
