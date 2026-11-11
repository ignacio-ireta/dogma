-- Wrong on purpose: codes come out in descending order instead of ascending.
SELECT warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION
SELECT warehouse_code
FROM returns
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code DESC;
