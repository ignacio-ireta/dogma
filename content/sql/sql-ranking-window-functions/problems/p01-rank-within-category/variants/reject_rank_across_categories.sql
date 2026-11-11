-- Wrong on purpose: RANK ignores the category, so its ranking runs across all categories.
SELECT
    category,
    product_id,
    revenue,
    ROW_NUMBER() OVER (
        PARTITION BY category
        ORDER BY revenue DESC NULLS LAST, product_id ASC
    ) AS row_number,
    RANK() OVER (
        ORDER BY revenue DESC NULLS LAST
    ) AS rank,
    DENSE_RANK() OVER (
        PARTITION BY category
        ORDER BY revenue DESC NULLS LAST
    ) AS dense_rank
FROM products
ORDER BY category ASC, row_number ASC;
