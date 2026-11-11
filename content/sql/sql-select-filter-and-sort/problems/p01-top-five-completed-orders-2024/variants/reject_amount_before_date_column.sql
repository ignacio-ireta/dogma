-- Wrong on purpose: the output columns come out in the wrong order, with amount before order_date.
SELECT
    order_id,
    customer_id,
    amount,
    order_date
FROM orders
WHERE status = 'completed'
  AND order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2025-01-01'
  AND amount IS NOT NULL
ORDER BY amount DESC, order_id ASC
LIMIT 5;
