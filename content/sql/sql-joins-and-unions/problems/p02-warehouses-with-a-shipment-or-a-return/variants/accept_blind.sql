WITH all_codes AS (
    SELECT warehouse_code AS warehouse_code FROM shipments
    UNION
    SELECT warehouse_code FROM returns
)
SELECT warehouse_code
FROM all_codes
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code ASC;
