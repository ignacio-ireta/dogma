-- The window is a calendar interval, so it is a RANGE frame on the date, not a ROWS frame.
-- RANGE BETWEEN INTERVAL 6 DAYS PRECEDING AND CURRENT ROW holds every row whose date lies
-- between (current date - 6 days) and the current date. Closed days have no rows, so they
-- shift nothing and are not counted.
--
-- PARTITION BY store_id keeps the windows of different shops apart.
-- Dates are unique within a store, so no two rows are peers and the RANGE frame has no
-- peer surprises (see the lesson on the default frame).
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
ORDER BY store_id, sales_date;
