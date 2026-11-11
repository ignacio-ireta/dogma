-- Blind solution, written from problem.md alone.
WITH months AS (
    SELECT DISTINCT user_id, CAST(DATE_TRUNC('month', event_at) AS DATE) AS m
    FROM activity
),
firsts AS (
    SELECT user_id, MIN(m) AS cohort_month FROM months GROUP BY user_id
),
sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size FROM firsts GROUP BY cohort_month
),
latest AS (
    SELECT MAX(m) AS last_m FROM months
),
grid AS (
    SELECT s.cohort_month, s.cohort_size, g.p AS period
    FROM sizes AS s
    CROSS JOIN latest AS l
    CROSS JOIN generate_series(0, 1200) AS g(p)
    WHERE g.p <= DATE_DIFF('month', s.cohort_month, l.last_m)
),
counts AS (
    SELECT
        f.cohort_month,
        DATE_DIFF('month', f.cohort_month, mo.m) AS period,
        COUNT(DISTINCT mo.user_id) AS active_users
    FROM months AS mo
    JOIN firsts AS f USING (user_id)
    GROUP BY f.cohort_month, period
)
SELECT
    g.cohort_month,
    g.period,
    CAST(COALESCE(c.active_users, 0) AS BIGINT) AS active_users,
    CAST(g.cohort_size AS BIGINT) AS cohort_size,
    CAST(ROUND(COALESCE(c.active_users, 0) / g.cohort_size, 4) AS DOUBLE) AS retention_rate
FROM grid AS g
LEFT JOIN counts AS c ON c.cohort_month = g.cohort_month AND c.period = g.period
ORDER BY g.cohort_month, g.period;
