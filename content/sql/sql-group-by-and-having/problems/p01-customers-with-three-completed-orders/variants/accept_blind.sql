-- Blind: filter first in a CTE, aggregate, then keep groups with at least three rows.
WITH done AS (
    SELECT customer_id, amount
    FROM orders
    WHERE status = 'completed'
),
per_customer AS (
    SELECT
        customer_id,
        COUNT(*) AS completed_orders,
        SUM(amount) AS total_amount
    FROM done
    GROUP BY customer_id
)
SELECT customer_id, completed_orders, total_amount
FROM per_customer
WHERE completed_orders >= 3
ORDER BY total_amount DESC NULLS LAST, customer_id ASC;
