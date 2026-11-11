-- Wrong on purpose: uses RANK, so the rank after a tie skips a number and a tie at the cut is cut off.
SELECT category, product_id, revenue, dense_rank
FROM (
    SELECT
        category,
        product_id,
        revenue,
        RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST
        ) AS dense_rank
    FROM products
) AS ranked
WHERE dense_rank <= 2
ORDER BY category ASC, dense_rank ASC, product_id ASC;
