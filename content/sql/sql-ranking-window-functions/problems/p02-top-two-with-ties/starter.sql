-- Problem: per category, every product whose DENSE_RANK by revenue is 1 or 2 (ties kept).
--
-- Tables:
--   products
--     product_id  BIGINT    -- unique
--     category    VARCHAR
--     revenue     DOUBLE    -- may be NULL
--
-- DENSE_RANK inside each category, revenue descending, NULL revenue last (NULLs are peers).
-- Keep the products with dense_rank <= 2.
-- Output columns, in this order: category, product_id, revenue, dense_rank.
-- Order the rows by category ascending, then dense_rank ascending, then product_id ascending.
--
-- Replace the line below with your query.

SELECT 1;
