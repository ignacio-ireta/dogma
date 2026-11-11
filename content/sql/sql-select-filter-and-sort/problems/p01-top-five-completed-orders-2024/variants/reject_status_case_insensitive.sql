-- Wrong on purpose: status is matched without regard to capitalisation, so Completed and COMPLETED also qualify.
SELECT
    order_id,
    customer_id,
    order_date,
    amount
FROM orders
WHERE lower(status) = 'completed'
  AND order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2025-01-01'
  AND amount IS NOT NULL
ORDER BY amount DESC, order_id ASC
LIMIT 5;
