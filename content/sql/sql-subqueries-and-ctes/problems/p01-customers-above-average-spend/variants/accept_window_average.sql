-- Correct solution: the average is a window aggregate over the buyer rows, not a second CTE.
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
with_average AS (
    SELECT
        customer_id,
        region,
        total_spend,
        AVG(total_spend) OVER () AS average_spend
    FROM buyer_totals
)
SELECT customer_id, region, total_spend, average_spend
FROM with_average
WHERE total_spend > average_spend
ORDER BY total_spend DESC, customer_id ASC;
