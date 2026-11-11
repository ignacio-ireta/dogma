-- Correct solution: the customer average is a window aggregate partitioned by customer.
SELECT
    order_id,
    customer_id,
    amount,
    customer_average
FROM (
    SELECT
        o.order_id,
        o.customer_id,
        o.amount,
        o.status,
        AVG(CASE WHEN o.status = 'completed' THEN o.amount END)
            OVER (PARTITION BY o.customer_id) AS customer_average
    FROM orders AS o
) AS scored
WHERE status = 'completed'
  AND customer_id IS NOT NULL
  AND amount IS NOT NULL
  AND amount > customer_average
ORDER BY order_id ASC;
