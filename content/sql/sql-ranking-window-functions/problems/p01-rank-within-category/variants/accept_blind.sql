SELECT
    category,
    product_id,
    revenue,
    ROW_NUMBER() OVER w_row AS row_number,
    RANK() OVER w_rank AS rank,
    DENSE_RANK() OVER w_dense AS dense_rank
FROM products
WINDOW
    w_row AS (PARTITION BY category ORDER BY revenue DESC NULLS LAST, product_id ASC),
    w_rank AS (PARTITION BY category ORDER BY revenue DESC NULLS LAST),
    w_dense AS (PARTITION BY category ORDER BY revenue DESC NULLS LAST)
ORDER BY category ASC, row_number ASC;
