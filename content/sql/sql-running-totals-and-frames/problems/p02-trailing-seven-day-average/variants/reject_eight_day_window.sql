-- Wrong on purpose: the window reaches seven days back, so it covers eight calendar days.
SELECT
    store_id,
    sales_date,
    AVG(revenue_cents) OVER (
        PARTITION BY store_id
        ORDER BY sales_date
        RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW
    ) AS trailing_average_cents,
    COUNT(*) OVER (
        PARTITION BY store_id
        ORDER BY sales_date
        RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW
    ) AS days_in_window
FROM daily_sales
ORDER BY store_id, sales_date;
