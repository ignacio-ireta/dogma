-- Wrong on purpose: orders of customers missing from the customer table are added as extra rows.
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
),
unknown AS (
    SELECT
        o.customer_id,
        CAST(NULL AS VARCHAR) AS customer_name,
        COUNT(*) AS order_count,
        SUM(o.amount) AS total_amount
    FROM orders AS o
    WHERE o.customer_id NOT IN (SELECT customer_id FROM customers)
    GROUP BY o.customer_id
)
SELECT * FROM base
UNION ALL
SELECT * FROM unknown
ORDER BY total_amount DESC, customer_id ASC;
