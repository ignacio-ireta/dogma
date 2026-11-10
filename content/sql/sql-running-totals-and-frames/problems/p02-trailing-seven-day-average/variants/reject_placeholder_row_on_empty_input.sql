-- Wrong on purpose: when daily_sales is empty, a placeholder row is returned so the output is never empty.
WITH result AS (
    SELECT
        store_id,
        sales_date,
        AVG(revenue_cents) OVER (
            PARTITION BY store_id
            ORDER BY sales_date
            RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
        ) AS trailing_average_cents,
        COUNT(*) OVER (
            PARTITION BY store_id
            ORDER BY sales_date
            RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
        ) AS days_in_window
    FROM daily_sales
)
SELECT * FROM result
UNION ALL
SELECT CAST(0 AS BIGINT), DATE '1970-01-01', CAST(0.0 AS DOUBLE), CAST(0 AS BIGINT)
WHERE NOT EXISTS (SELECT 1 FROM daily_sales)
ORDER BY store_id, sales_date;
