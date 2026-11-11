-- Blind: month calendar as a CTE; each event assigned to a month by truncation, then joined.
WITH calendar AS (
    SELECT CAST(g AS DATE) AS month
    FROM generate_series(TIMESTAMP '2023-11-01', TIMESTAMP '2024-02-01', INTERVAL 1 MONTH) AS t(g)
),
bucketed AS (
    SELECT user_id, CAST(DATE_TRUNC('month', event_ts) AS DATE) AS month
    FROM events
    WHERE event_ts >= TIMESTAMP '2023-11-01' AND event_ts < TIMESTAMP '2024-03-01'
),
counts AS (
    SELECT month, COUNT(DISTINCT user_id) AS active_users
    FROM bucketed
    GROUP BY month
)
SELECT c.month, COALESCE(k.active_users, 0)::BIGINT AS active_users
FROM calendar AS c
LEFT JOIN counts AS k ON k.month = c.month
ORDER BY c.month;
