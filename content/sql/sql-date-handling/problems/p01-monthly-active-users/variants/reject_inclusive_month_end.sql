-- Wrong on purpose: an event exactly at midnight on the first of the next month is also counted in the month
--
-- Grain first: the answer has one row per month of the range, so the calendar is the left side of
-- the join. Events can only add to a month, never create one. A month with no events still gets a
-- row, with a count of 0.
--
-- A month is a half-open interval: an event belongs to the month when
-- event_ts >= first instant of the month AND event_ts < first instant of the next month.
-- BETWEEN would also include the next month's first instant (midnight on the 1st), and "< last day"
-- would drop every event on the last day of the month.
WITH months AS (
    -- generate_series includes its stop value, so the last month of the range is kept.
    -- The range comes from the statement, not from MIN and MAX of the events.
    SELECT CAST(g AS DATE) AS month
    FROM generate_series(DATE '2023-11-01', DATE '2024-02-01', INTERVAL 1 MONTH) AS t(g)
)
SELECT
    m.month,
    COUNT(DISTINCT e.user_id) AS active_users   -- distinct users: several events in a month count once
FROM months AS m
LEFT JOIN events AS e
    ON e.event_ts >= m.month
   AND e.event_ts <= m.month + INTERVAL 1 MONTH
GROUP BY m.month
ORDER BY m.month;
