-- Blind: date minus row number over distinct days, grouped per user.
WITH days AS (
    SELECT DISTINCT user_id, visit_date FROM visits
),
keyed AS (
    SELECT user_id, visit_date,
           visit_date - CAST(ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY visit_date) AS INTEGER) AS k
    FROM days
)
SELECT user_id,
       MIN(visit_date) AS island_start,
       MAX(visit_date) AS island_end,
       CAST(COUNT(*) AS BIGINT) AS island_days
FROM keyed
GROUP BY user_id, k
ORDER BY user_id, island_start;
