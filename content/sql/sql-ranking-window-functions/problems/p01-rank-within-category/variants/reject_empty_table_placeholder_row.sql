-- Wrong on purpose: adds a placeholder row when products is empty, so an empty table returns one row.
SELECT category, product_id, revenue, row_number, rank, dense_rank
FROM (
    SELECT
        category,
        product_id,
        revenue,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST, product_id ASC
        ) AS row_number,
        RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST
        ) AS rank,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST
        ) AS dense_rank
    FROM products
    UNION ALL
    SELECT 'none', CAST(NULL AS BIGINT), CAST(NULL AS DOUBLE), 1, 1, 1
    WHERE NOT EXISTS (SELECT 1 FROM products)
)
ORDER BY category ASC, row_number ASC;
