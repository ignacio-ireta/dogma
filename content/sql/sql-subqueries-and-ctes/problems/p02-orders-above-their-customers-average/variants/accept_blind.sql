WITH avgs AS (
    SELECT customer_id, AVG(amount) AS customer_average
    FROM orders
    WHERE status = 'completed' AND customer_id IS NOT NULL
    GROUP BY customer_id
)
SELECT o.order_id, o.customer_id, o.amount, a.customer_average
FROM orders o
JOIN avgs a ON a.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.amount IS NOT NULL AND o.amount > a.customer_average
ORDER BY o.order_id
