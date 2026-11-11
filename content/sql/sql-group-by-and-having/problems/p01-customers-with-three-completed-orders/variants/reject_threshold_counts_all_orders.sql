-- Wrong on purpose: the threshold counts every order of the customer, not only completed ones.
--
-- Reference query: customers with three or more completed orders.
SELECT
    customer_id,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_orders,
    SUM(amount) FILTER (WHERE status = 'completed') AS total_amount
FROM orders
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
