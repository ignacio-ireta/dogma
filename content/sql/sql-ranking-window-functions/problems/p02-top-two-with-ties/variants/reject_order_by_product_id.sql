-- Wrong on purpose: sorts the rows by product_id inside each category, not by dense_rank.
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
WHERE dense_rank <= 2
ORDER BY category ASC, product_id ASC;
