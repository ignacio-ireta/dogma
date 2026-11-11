WITH totals AS (
    SELECT c.customer_id, c.region, SUM(o.amount) AS total_spend
    FROM customers c
    JOIN orders o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.region
),
avg_t AS (SELECT AVG(total_spend) AS average_spend FROM totals)
SELECT t.customer_id, t.region, t.total_spend, a.average_spend
FROM totals t CROSS JOIN avg_t a
WHERE t.total_spend > a.average_spend
ORDER BY t.total_spend DESC, t.customer_id ASC
