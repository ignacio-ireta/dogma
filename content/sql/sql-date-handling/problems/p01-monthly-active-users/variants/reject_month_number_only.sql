-- Wrong on purpose: a month is matched by its calendar number only, so the same month of another year is counted
WITH months AS (
    SELECT CAST(g AS DATE) AS month
    FROM generate_series(DATE '2023-11-01', DATE '2024-02-01', INTERVAL 1 MONTH) AS t(g)
)
SELECT
    m.month,
    COUNT(DISTINCT e.user_id) AS active_users
FROM months AS m
LEFT JOIN events AS e
    ON EXTRACT(month FROM e.event_ts) = EXTRACT(month FROM m.month)
GROUP BY m.month
ORDER BY m.month;
