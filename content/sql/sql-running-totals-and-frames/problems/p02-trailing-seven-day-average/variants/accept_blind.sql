SELECT
    store_id,
    sales_date,
    AVG(revenue_cents) OVER w AS trailing_average_cents,
    COUNT(*) OVER w AS days_in_window
FROM daily_sales
WINDOW w AS (
    PARTITION BY store_id
    ORDER BY sales_date
    RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW
)
ORDER BY store_id, sales_date;
