SELECT order_id, customer_id, order_date, amount
FROM orders
WHERE status = 'completed'
  AND year(order_date) = 2024
  AND amount IS NOT NULL
ORDER BY amount DESC, order_id
LIMIT 5;
