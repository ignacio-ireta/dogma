-- Wrong on purpose: partitions by a lower-cased category, so case variants share one dense rank.
SELECT category, product_id, revenue, dense_rank
FROM (
    SELECT
        category,
        product_id,
        revenue,
        DENSE_RANK() OVER (
            PARTITION BY lower(category)
            ORDER BY revenue DESC NULLS LAST
        ) AS dense_rank
    FROM products
) AS ranked
WHERE dense_rank <= 2
ORDER BY category ASC, dense_rank ASC, product_id ASC;
