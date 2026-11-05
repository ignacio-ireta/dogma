-- Problem: warehouses with a shipment or a return.
--
-- Tables:
--   shipments
--     shipment_id     BIGINT
--     warehouse_code  VARCHAR   -- may be NULL
--     shipped_on      DATE
--   returns
--     return_id       BIGINT
--     warehouse_code  VARCHAR   -- may be NULL
--     returned_on     DATE
--
-- Return one column, warehouse_code: each distinct non-NULL code that appears in shipments,
-- in returns, or in both. Codes are exact (no UPPER, LOWER or TRIM). Sort ascending.
--
-- Replace the line below with your query.

SELECT 1;
