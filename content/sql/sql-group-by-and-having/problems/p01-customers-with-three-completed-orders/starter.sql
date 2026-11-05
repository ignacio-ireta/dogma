-- Problem: customers with three or more completed orders.
--
-- Table: orders
--   order_id     BIGINT    -- unique order id
--   customer_id  BIGINT
--   order_date   DATE
--   status       VARCHAR   -- e.g. 'completed', 'pending', 'cancelled', 'refunded'
--   amount       DOUBLE    -- may be NULL
--
-- For each customer with at least three completed orders, return:
--   customer_id       the customer
--   completed_orders  how many completed orders they have (a NULL amount still counts)
--   total_amount      the sum of their completed amounts (NULL amounts are skipped; NULL if
--                     every amount is NULL)
-- Order by total_amount descending, NULLs last, then customer_id ascending.
--
-- Replace the line below with your query.

SELECT 1;
