-- Reference query written as a common table expression, then selected in the output order.
WITH ranked AS (
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
)
SELECT category, product_id, revenue, row_number, rank, dense_rank
FROM ranked
ORDER BY category ASC, row_number ASC;
