-- Wrong on purpose: uses ROW_NUMBER, so exactly two rows per category are kept and ties at the cut are split.
SELECT category, product_id, revenue, dense_rank
FROM (
    SELECT
        category,
        product_id,
        revenue,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST, product_id ASC
        ) AS dense_rank
    FROM products
) AS ranked
WHERE dense_rank <= 2
ORDER BY category ASC, dense_rank ASC, product_id ASC;
