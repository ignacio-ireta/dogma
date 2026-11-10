-- Correct solution: the trailing window is a self-join on a seven-day date range, then grouped.
SELECT
    d.store_id,
    d.sales_date,
    AVG(w.revenue_cents) AS trailing_average_cents,
    COUNT(*) AS days_in_window
FROM daily_sales AS d
JOIN daily_sales AS w
  ON w.store_id = d.store_id
 AND w.sales_date BETWEEN d.sales_date - INTERVAL 6 DAYS AND d.sales_date
GROUP BY d.store_id, d.sales_date
ORDER BY d.store_id, d.sales_date;
