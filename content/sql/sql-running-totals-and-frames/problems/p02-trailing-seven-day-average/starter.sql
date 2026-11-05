-- daily_sales(store_id BIGINT, sales_date DATE, revenue_cents BIGINT)
-- For each row, the mean revenue over the seven calendar days ending on its date
-- (sales_date - 6 days through sales_date, both included), within its store.
-- Output columns: store_id BIGINT, sales_date DATE, trailing_average_cents DOUBLE,
--   days_in_window BIGINT. Order by store_id, sales_date.
SELECT 1;
