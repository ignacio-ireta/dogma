-- Wrong on purpose: the final ORDER BY sorts by product_id, not by row_number within each category.
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
ORDER BY category ASC, product_id ASC;
