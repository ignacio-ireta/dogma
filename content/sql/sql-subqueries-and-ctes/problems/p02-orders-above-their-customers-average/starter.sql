-- Problem: completed orders whose amount is above their own customer's average.
--
-- Table: orders
--   order_id     BIGINT    -- unique order id
--   customer_id  BIGINT    -- may be NULL: such an order belongs to no customer
--   order_date   DATE
--   status       VARCHAR   -- e.g. 'completed', 'pending', 'cancelled', 'refunded'
--   amount       DOUBLE    -- may be NULL
--
-- A candidate: status exactly 'completed', amount not NULL, customer_id not NULL.
-- customer_average: the average amount of that customer's completed orders (NULLs skipped).
-- Return candidates whose amount is strictly above customer_average, with these columns:
--   order_id, customer_id, amount, customer_average
-- Order by order_id ascending.
--
-- Replace the line below with your query.

SELECT 1;
