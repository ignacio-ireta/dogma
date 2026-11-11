-- Correct, differently written: FILTER clauses in place of a WHERE, with the threshold
-- counted over completed orders only.
SELECT
    customer_id,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_orders,
    SUM(amount) FILTER (WHERE status = 'completed') AS total_amount
FROM orders
GROUP BY customer_id
HAVING COUNT(*) FILTER (WHERE status = 'completed') >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
