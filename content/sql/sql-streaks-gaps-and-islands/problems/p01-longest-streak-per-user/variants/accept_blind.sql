-- Blind: LAG flags on distinct days, a running count of starts, then the longest run per user.
WITH days AS (
    SELECT DISTINCT user_id, visit_date FROM visits
),
flags AS (
    SELECT user_id, visit_date,
           CASE WHEN LAG(visit_date) OVER (PARTITION BY user_id ORDER BY visit_date) IS NULL
                  OR visit_date - LAG(visit_date) OVER (PARTITION BY user_id ORDER BY visit_date) > 1
                THEN 1 ELSE 0 END AS starts
    FROM days
),
runs AS (
    SELECT user_id, visit_date,
           SUM(starts) OVER (PARTITION BY user_id ORDER BY visit_date) AS run_no
    FROM flags
),
run_stats AS (
    SELECT user_id, run_no,
           MIN(visit_date) AS streak_start,
           MAX(visit_date) AS streak_end,
           COUNT(*) AS streak_days
    FROM runs
    GROUP BY user_id, run_no
)
SELECT user_id,
       CAST(streak_days AS BIGINT) AS streak_days,
       streak_start,
       streak_end
FROM (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY user_id ORDER BY streak_days DESC, streak_start
    ) AS pos
    FROM run_stats
) AS best
WHERE pos = 1
ORDER BY user_id;
