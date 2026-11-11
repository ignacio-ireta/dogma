-- Wrong on purpose: when customers is empty, a placeholder row of zeros is returned.
WITH base AS (
    SELECT
        c.customer_id,
        c.customer_name,
        COUNT(o.order_id) AS order_count,
        COALESCE(SUM(o.amount), 0.0) AS total_amount
    FROM customers AS c
    LEFT JOIN orders AS o
        ON o.customer_id = c.customer_id
    WHERE c.is_current
    GROUP BY c.customer_id, c.customer_name
)
SELECT * FROM base
UNION ALL
SELECT
    CAST(NULL AS BIGINT),
    CAST(NULL AS VARCHAR),
    CAST(0 AS BIGINT),
    0.0
WHERE NOT EXISTS (SELECT 1 FROM customers)
ORDER BY total_amount DESC, customer_id ASC;
