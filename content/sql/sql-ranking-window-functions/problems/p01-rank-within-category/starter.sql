-- Problem: rank every product inside its category by revenue, three ways.
--
-- Tables:
--   products
--     product_id  BIGINT    -- unique
--     category    VARCHAR
--     revenue     DOUBLE    -- may be NULL
--
-- Output columns, in this order: category, product_id, revenue, row_number, rank, dense_rank.
--   row_number : ROW_NUMBER, revenue descending, NULL revenue last, ties broken by product_id ascending
--   rank       : RANK, revenue descending, NULL revenue last (ties share a rank, the next rank skips)
--   dense_rank : DENSE_RANK, same order as rank (ties share a rank, no gaps)
-- Every product appears once. Order the rows by category ascending, then row_number ascending.
--
-- Replace the line below with your query.

SELECT 1;
