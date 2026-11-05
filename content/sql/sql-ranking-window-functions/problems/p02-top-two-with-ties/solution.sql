-- Reference query: the products whose DENSE_RANK in their category is at most 2.
--
-- A window function cannot be used in WHERE, so the rank is computed in a derived table first and
-- the filter is applied outside it. DENSE_RANK keeps ties as peers and does not skip a rank, so a tie
-- at the cut brings in every tied product. NULLS LAST puts a NULL revenue after every number.
SELECT
    category,
    product_id,
    revenue,
    dense_rank
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
ORDER BY category ASC, dense_rank ASC, product_id ASC;
