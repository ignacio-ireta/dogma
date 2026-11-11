-- Wrong on purpose: equal totals are ordered by customer_id descending.
--
-- Reference query: customers with three or more completed orders.
SELECT
    customer_id,
    COUNT(*) AS completed_orders,
    SUM(amount) AS total_amount
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id DESC;
