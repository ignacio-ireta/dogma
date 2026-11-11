-- Wrong on purpose: puts NULL revenue first, so NULL products take the top dense ranks.
SELECT category, product_id, revenue, dense_rank
FROM (
    SELECT
        category,
        product_id,
        revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS FIRST
        ) AS dense_rank
    FROM products
) AS ranked
WHERE dense_rank <= 2
ORDER BY category ASC, dense_rank ASC, product_id ASC;
