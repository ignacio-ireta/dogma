-- Wrong on purpose: total_amount is returned before completed_orders.
--
-- Reference query: customers with three or more completed orders.
SELECT
    customer_id,
    SUM(amount) AS total_amount,
    COUNT(*) AS completed_orders
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
