-- Wrong on purpose: breaks ROW_NUMBER ties by product_id as text, so 905113208 sorts before 92.
SELECT
    category,
    product_id,
    revenue,
    ROW_NUMBER() OVER (
        PARTITION BY category
        ORDER BY revenue DESC NULLS LAST, CAST(product_id AS VARCHAR) ASC
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
ORDER BY category ASC, row_number ASC;
