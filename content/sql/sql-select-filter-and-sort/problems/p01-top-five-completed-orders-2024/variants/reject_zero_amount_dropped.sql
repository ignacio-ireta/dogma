-- Wrong on purpose: a zero amount is treated as unknown and dropped, so zero-amount orders never qualify.
SELECT
    order_id,
    customer_id,
    order_date,
    amount
FROM orders
WHERE status = 'completed'
  AND order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2025-01-01'
  AND amount > 0
ORDER BY amount DESC, order_id ASC
LIMIT 5;
