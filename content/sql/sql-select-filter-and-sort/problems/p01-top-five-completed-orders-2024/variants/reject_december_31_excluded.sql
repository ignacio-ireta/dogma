-- Wrong on purpose: the end of the 2024 range is exclusive at 2024-12-31, so orders placed on that day are dropped.
SELECT
    order_id,
    customer_id,
    order_date,
    amount
FROM orders
WHERE status = 'completed'
  AND order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2024-12-31'
  AND amount IS NOT NULL
ORDER BY amount DESC, order_id ASC
LIMIT 5;
