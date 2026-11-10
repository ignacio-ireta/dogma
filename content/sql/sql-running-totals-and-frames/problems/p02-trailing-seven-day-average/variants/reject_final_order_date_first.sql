-- Wrong on purpose: the final order sorts by date first, so stores interleave.
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
ORDER BY sales_date, store_id;
