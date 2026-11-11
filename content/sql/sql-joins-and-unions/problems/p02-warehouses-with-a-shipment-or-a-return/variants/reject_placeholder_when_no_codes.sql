-- Wrong on purpose: when no table has a code, a placeholder row with the text none is returned.
SELECT warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION
SELECT warehouse_code
FROM returns
WHERE warehouse_code IS NOT NULL
UNION
SELECT 'none' AS warehouse_code
WHERE NOT EXISTS (SELECT 1 FROM shipments WHERE warehouse_code IS NOT NULL)
  AND NOT EXISTS (SELECT 1 FROM returns WHERE warehouse_code IS NOT NULL)
ORDER BY warehouse_code ASC;
