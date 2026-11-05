-- Step 1 (distinct days): duplicate visits on one day are one active day.
WITH distinct_days AS (
    SELECT DISTINCT user_id, visit_date
    FROM visits
),
-- Step 2 (gap flags): LAG gives the previous active day of the same user. A difference of exactly
-- one day means the run continues (flag 0). Any other difference, and the first day of each user
-- (LAG is NULL there, so the comparison is not true), starts a new streak (flag 1).
flagged AS (
    SELECT
        user_id,
        visit_date,
        CASE
            WHEN visit_date - LAG(visit_date) OVER (
                PARTITION BY user_id ORDER BY visit_date
            ) = 1 THEN 0
            ELSE 1
        END AS new_streak
    FROM distinct_days
),
-- Step 3 (streak id): a running SUM of the flags counts the streaks so far for each user. Every
-- day of one streak gets the same running total, so the total works as a streak id.
with_id AS (
    SELECT
        user_id,
        visit_date,
        SUM(new_streak) OVER (
            PARTITION BY user_id
            ORDER BY visit_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS streak_no
    FROM flagged
)
-- Step 4 (one row per streak): the ends are MIN and MAX of its dates, and COUNT is its length.
SELECT
    user_id,
    MIN(visit_date) AS island_start,
    MAX(visit_date) AS island_end,
    CAST(COUNT(*) AS BIGINT) AS island_days
FROM with_id
GROUP BY user_id, streak_no
ORDER BY user_id ASC, island_start ASC;
