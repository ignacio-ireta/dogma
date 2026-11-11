-- Wrong on purpose: equal amounts are ordered by order_id descending instead of ascending.
SELECT
    order_id,
    customer_id,
    order_date,
    amount
FROM orders
WHERE status = 'completed'
  AND order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2025-01-01'
  AND amount IS NOT NULL
ORDER BY amount DESC, order_id DESC
LIMIT 5;
