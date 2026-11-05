-- Reference query: customers with three or more completed orders.
--
-- Logical order: FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY.
-- WHERE removes non-completed rows BEFORE grouping, so they never reach the count.
-- HAVING filters whole groups AFTER aggregation, so it can test COUNT(*).
SELECT
    customer_id,
    COUNT(*) AS completed_orders,      -- counts rows, including those with NULL amount
    SUM(amount) AS total_amount        -- SUM skips NULL; all-NULL group gives NULL
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
