-- Wrong on purpose: the window is six preceding rows, so closed days are skipped as if they were days.
SELECT
    store_id,
    sales_date,
    AVG(revenue_cents) OVER (
        PARTITION BY store_id
        ORDER BY sales_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS trailing_average_cents,
    COUNT(*) OVER (
        PARTITION BY store_id
        ORDER BY sales_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS days_in_window
FROM daily_sales
ORDER BY store_id, sales_date;
