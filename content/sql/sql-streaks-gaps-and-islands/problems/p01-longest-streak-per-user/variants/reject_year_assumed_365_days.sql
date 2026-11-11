-- Wrong on purpose: Day keys assume every year has 365 days, so a run breaks at a leap year end.
-- Step 1 (remove duplicates): one row per (user, day). Without DISTINCT, two visits on one day
-- break the trick in step 2, because the row number moves while the date does not.
WITH distinct_days AS (
    SELECT DISTINCT user_id, visit_date
    FROM visits
),
-- Step 2 (the island key): within a user, the k-th distinct day is k days after the first day
-- of its run only while the run continues. Subtracting the row number from the date therefore
-- gives one constant value per run: consecutive days share the same key, and a gap changes it.
-- DATE minus INTEGER is a DATE in DuckDB, so the key is a date that means nothing by itself.
numbered AS (
    SELECT
        user_id,
        visit_date,
        CAST(EXTRACT(YEAR FROM visit_date) AS BIGINT) * 365
        + CAST(EXTRACT(DOY FROM visit_date) AS BIGINT)
        - ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY visit_date) AS island_key
    FROM distinct_days
),
-- Step 3 (one row per run): group by the key. Its MIN and MAX dates are the run's ends, and
-- COUNT is the number of active days in it.
islands AS (
    SELECT
        user_id,
        island_key,
        MIN(visit_date) AS streak_start,
        MAX(visit_date) AS streak_end,
        COUNT(*) AS streak_days
    FROM numbered
    GROUP BY user_id, island_key
),
-- Step 4 (the longest run per user): longest first, earliest start on a tie, keep row 1.
ranked AS (
    SELECT
        user_id,
        streak_days,
        streak_start,
        streak_end,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY streak_days DESC, streak_start ASC
        ) AS rn
    FROM islands
)
SELECT
    user_id,
    CAST(streak_days AS BIGINT) AS streak_days,
    streak_start,
    streak_end
FROM ranked
WHERE rn = 1
ORDER BY user_id ASC;
