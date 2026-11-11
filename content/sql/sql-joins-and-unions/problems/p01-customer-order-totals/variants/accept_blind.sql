-- Blind solution, written from problem.md alone: aggregate orders first, then attach them.
WITH per_customer AS (
    SELECT customer_id, COUNT(*) AS n, SUM(amount) AS total
    FROM orders
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    c.customer_name,
    COALESCE(p.n, 0) AS order_count,
    COALESCE(p.total, 0.0) AS total_amount
FROM customers AS c
LEFT JOIN per_customer AS p ON p.customer_id = c.customer_id
WHERE c.is_current
ORDER BY total_amount DESC, c.customer_id ASC;
