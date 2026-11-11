-- Wrong on purpose: a customer whose completed amounts are all NULL gets a total of 0.
--
-- Reference query: customers with three or more completed orders.
SELECT
    customer_id,
    COUNT(*) AS completed_orders,
    COALESCE(SUM(amount), 0) AS total_amount
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
