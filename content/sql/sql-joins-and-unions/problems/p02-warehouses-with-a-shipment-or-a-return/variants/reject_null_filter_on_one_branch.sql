-- Wrong on purpose: the NULL filter is applied only to shipments, so NULL returns still appear.
SELECT warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION
SELECT warehouse_code
FROM returns
ORDER BY warehouse_code ASC;
