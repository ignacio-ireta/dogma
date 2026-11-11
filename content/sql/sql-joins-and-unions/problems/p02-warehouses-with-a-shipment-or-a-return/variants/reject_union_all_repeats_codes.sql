-- Wrong on purpose: UNION ALL keeps one row per shipment and per return, so shared codes repeat.
SELECT warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION ALL
SELECT warehouse_code
FROM returns
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code ASC;
