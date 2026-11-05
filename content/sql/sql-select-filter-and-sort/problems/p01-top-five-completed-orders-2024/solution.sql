-- Reference query: top five completed orders of 2024.
--
-- Logical order of a SELECT: FROM -> WHERE -> SELECT -> ORDER BY -> LIMIT.
-- WHERE keeps only the qualifying rows, so ORDER BY and LIMIT see nothing else.
SELECT
    order_id,
    customer_id,
    order_date,
    amount
FROM orders
WHERE status = 'completed'                 -- exact match; "not pending" would keep refunds
  AND order_date >= DATE '2024-01-01'      -- half-open range: 2024-01-01 is in ...
  AND order_date <  DATE '2025-01-01'      -- ... and 2025-01-01 is out
  AND amount IS NOT NULL                   -- NULL is unknown, so it is never ranked
ORDER BY amount DESC, order_id ASC         -- order_id breaks ties, so the cut is deterministic
LIMIT 5;
