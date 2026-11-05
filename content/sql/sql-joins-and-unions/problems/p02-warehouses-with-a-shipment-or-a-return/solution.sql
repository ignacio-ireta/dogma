-- Reference query: distinct warehouse codes from either source, sorted.
--
-- UNION stacks the two branches and removes duplicates across them and within each one.
-- UNION ALL would keep every row, so BOS-1 would appear once per shipment and return.
-- The NULL filter sits inside each branch, before the union, so an unassigned code is removed
-- whichever table it came from.
SELECT warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION
SELECT warehouse_code
FROM returns
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code ASC;
