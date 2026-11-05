-- Reference query: every product with three ranks inside its category.
--
-- The three window functions differ in one place only: the tie-break. ROW_NUMBER must give every
-- row a distinct number, so its ORDER BY adds product_id. RANK and DENSE_RANK must keep ties as
-- peers, so their ORDER BY stops at revenue. Adding product_id to them would make no two rows tie.
-- NULLS LAST puts a NULL revenue after every number; NULLs are peers of each other.
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
ORDER BY category ASC, row_number ASC;
