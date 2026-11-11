-- Wrong on purpose: returns a placeholder row when no product qualifies, so an empty result is one row.
WITH ranked AS (
    SELECT
        category,
        product_id,
        revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST
        ) AS dense_rank
    FROM products
),
kept AS (
    SELECT category, product_id, revenue, dense_rank
    FROM ranked
    WHERE dense_rank <= 2
)
SELECT category, product_id, revenue, dense_rank
FROM kept
UNION ALL
SELECT 'none', CAST(NULL AS BIGINT), CAST(NULL AS DOUBLE), 1
WHERE NOT EXISTS (SELECT 1 FROM kept)
ORDER BY category ASC, dense_rank ASC, product_id ASC;
