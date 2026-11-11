-- Wrong on purpose: the rate is rounded to two decimal places instead of four.
-- "month" is one of these dates, so period offsets are whole calendar months and no day counts.
WITH events AS (
    SELECT
        user_id,
        CAST(DATE_TRUNC('month', event_at) AS DATE) AS activity_month
    FROM activity
),
-- Step 2 (cohort): each user's cohort is the earliest activity month. The cohort is set once per
-- user, from the first event, so a user with many events still has exactly one cohort.
first_month AS (
    SELECT user_id, MIN(activity_month) AS cohort_month
    FROM events
    GROUP BY user_id
),
-- Step 3 (denominator): the cohort size is the count of users in each cohort. It is fixed for the
-- whole row of the table, so every period of a cohort divides by the same N_c.
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM first_month
    GROUP BY cohort_month
),
-- Step 4 (numerator): distinct active users per cohort and period. DATE_DIFF('month') gives the
-- period offset, and COUNT(DISTINCT user_id) counts a user once however many events they logged.
active AS (
    SELECT
        f.cohort_month,
        DATE_DIFF('month', f.cohort_month, e.activity_month) AS period,
        COUNT(DISTINCT e.user_id) AS active_users
    FROM events AS e
    JOIN first_month AS f USING (user_id)
    GROUP BY f.cohort_month, period
),
-- Step 5 (the latest month): the right edge of the grid, the last activity month in the table.
horizon AS (
    SELECT CAST(DATE_TRUNC('month', MAX(event_at)) AS DATE) AS last_month
    FROM activity
),
-- Step 6 (the grid): every cohort gets periods 0..(months from cohort to the last month). Periods
-- with no activity are rows here, so the join below cannot drop them.
grid AS (
    SELECT
        cs.cohort_month,
        UNNEST(generate_series(0, DATE_DIFF('month', cs.cohort_month, h.last_month))) AS period
    FROM cohort_sizes AS cs
    CROSS JOIN horizon AS h
)
-- Step 7 (the rate): LEFT JOIN the counts onto the grid and fill empty cells with zero. The rate is
-- active over the cohort size, rounded once, at the end.
SELECT
    g.cohort_month,
    g.period,
    COALESCE(a.active_users, 0) AS active_users,
    cs.cohort_size,
    ROUND(COALESCE(a.active_users, 0) * 1.0 / cs.cohort_size, 2) AS retention_rate
FROM grid AS g
JOIN cohort_sizes AS cs USING (cohort_month)
LEFT JOIN active AS a
    ON a.cohort_month = g.cohort_month AND a.period = g.period
ORDER BY g.cohort_month ASC, g.period ASC;
