-- Wrong on purpose: partitions by a lower-cased category, so case differences merge groups.
SELECT
    category,
    product_id,
    revenue,
    ROW_NUMBER() OVER (
        PARTITION BY lower(category)
        ORDER BY revenue DESC NULLS LAST, product_id ASC
    ) AS row_number,
    RANK() OVER (
        PARTITION BY lower(category)
        ORDER BY revenue DESC NULLS LAST
    ) AS rank,
    DENSE_RANK() OVER (
        PARTITION BY lower(category)
        ORDER BY revenue DESC NULLS LAST
    ) AS dense_rank
FROM products
ORDER BY category ASC, row_number ASC;
