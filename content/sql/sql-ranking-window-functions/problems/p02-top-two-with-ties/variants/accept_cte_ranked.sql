-- Reference query with the rank computed in a common table expression instead of a derived table.
WITH ranked AS (
    SELECT
        category,
        product_id,
        revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY revenue DESC NULLS LAST
        ) AS dense_rank
    FROM products
)
SELECT
    category,
    product_id,
    revenue,
    dense_rank
FROM ranked
WHERE dense_rank <= 2
ORDER BY category ASC, dense_rank ASC, product_id ASC;
