-- Wrong on purpose: days_in_window counts every calendar day since the store's first row, closed days included.
SELECT
    store_id,
    sales_date,
    AVG(revenue_cents) OVER (
        PARTITION BY store_id
        ORDER BY sales_date
        RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
    ) AS trailing_average_cents,
    LEAST(
        7,
        CAST(sales_date - MIN(sales_date) OVER (PARTITION BY store_id) AS BIGINT) + 1
    ) AS days_in_window
FROM daily_sales
ORDER BY store_id, sales_date;
