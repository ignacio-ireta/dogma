-- Wrong on purpose: stray spaces are trimmed, so a padded code merges with its clean spelling.
SELECT TRIM(warehouse_code) AS warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION
SELECT TRIM(warehouse_code)
FROM returns
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code ASC;
