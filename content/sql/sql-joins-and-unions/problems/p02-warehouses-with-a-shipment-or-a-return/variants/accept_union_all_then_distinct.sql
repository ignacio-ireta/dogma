-- Correct, different shape: stack both tables first, then remove duplicates with DISTINCT.
SELECT DISTINCT warehouse_code
FROM (
    SELECT warehouse_code FROM shipments
    UNION ALL
    SELECT warehouse_code FROM returns
) AS codes
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code ASC;
