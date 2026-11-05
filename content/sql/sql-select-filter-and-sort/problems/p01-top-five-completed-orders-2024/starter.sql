-- Problem: top five completed orders of 2024.
--
-- Table: orders
--   order_id     BIGINT    -- unique order id
--   customer_id  BIGINT
--   order_date   DATE      -- compare with DATE '2024-01-01' style literals
--   status       VARCHAR   -- e.g. 'completed', 'pending', 'cancelled', 'refunded', 'processing'
--   amount       DOUBLE    -- may be NULL
--
-- Return the five largest completed orders placed in 2024. Orders with a NULL amount do not
-- qualify. Columns, in this order: order_id, customer_id, order_date, amount.
-- Sort by amount descending, then order_id ascending. At most five rows.
--
-- Replace the line below with your query.

SELECT 1;
