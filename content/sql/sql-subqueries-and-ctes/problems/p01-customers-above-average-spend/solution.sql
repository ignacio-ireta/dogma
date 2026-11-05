-- Reference query: buyers whose total completed spend is above the average buyer total.
--
-- Two stages, each named by a common table expression (CTE):
--   1. buyer_totals: one row per buyer. Only completed orders of a customer that has a row in
--      customers count. The inner join drops NULL customer ids and ids with no customer row, so
--      those orders never reach the average. SUM skips NULL amounts, so a buyer whose amounts are
--      all NULL gets a NULL total.
--   2. average_total: AVG over the buyer totals. AVG skips NULL totals. A customer with no
--      completed order has no row in buyer_totals, so it is never in the average.
-- The final query compares each buyer with that one number. The comparison is strict (>), so a
-- buyer exactly at the average is not returned. A NULL total compares as unknown, so it is not
-- returned either.
WITH buyer_totals AS (
    SELECT
        o.customer_id,
        c.region,
        SUM(o.amount) AS total_spend
    FROM orders AS o
    INNER JOIN customers AS c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.region
),
average_total AS (
    -- An aggregate without GROUP BY always returns one row, even when buyer_totals is empty.
    SELECT AVG(total_spend) AS average_spend
    FROM buyer_totals
)
SELECT
    b.customer_id,
    b.region,
    b.total_spend,
    a.average_spend
FROM buyer_totals AS b
CROSS JOIN average_total AS a
WHERE b.total_spend > a.average_spend
ORDER BY b.total_spend DESC, b.customer_id ASC;
