-- Wrong on purpose: drops products with NULL revenue even when their dense rank is 1 or 2.
SELECT category, product_id, revenue, dense_rank
FROM (
    SELECT
        category,
        product_id,
        revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST
        ) AS dense_rank
    FROM products
) AS ranked
WHERE dense_rank <= 2 AND revenue IS NOT NULL
ORDER BY category ASC, dense_rank ASC, product_id ASC;
