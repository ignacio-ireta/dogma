-- Wrong on purpose: codes are upper-cased, so bos-1 merges into BOS-1.
SELECT UPPER(warehouse_code) AS warehouse_code
FROM shipments
WHERE warehouse_code IS NOT NULL
UNION
SELECT UPPER(warehouse_code)
FROM returns
WHERE warehouse_code IS NOT NULL
ORDER BY warehouse_code ASC;
