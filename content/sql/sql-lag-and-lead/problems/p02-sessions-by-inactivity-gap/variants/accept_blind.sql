-- Blind solution, written from problem.md alone: compare gaps in exact microseconds.
WITH ordered AS (
    SELECT
        visitor_id,
        view_id,
        viewed_at,
        LAG(viewed_at) OVER (PARTITION BY visitor_id ORDER BY viewed_at, view_id) AS prev_at
    FROM screen_views
),
marked AS (
    SELECT
        visitor_id,
        view_id,
        viewed_at,
        CASE
            WHEN prev_at IS NULL THEN 1
            WHEN epoch_us(viewed_at) - epoch_us(prev_at) > 1800000000 THEN 1
            ELSE 0
        END AS start_flag
    FROM ordered
)
SELECT
    visitor_id,
    view_id,
    viewed_at,
    CAST(SUM(start_flag) OVER (PARTITION BY visitor_id ORDER BY viewed_at, view_id) AS BIGINT) AS session_number,
    start_flag = 1 AS starts_session
FROM marked
ORDER BY visitor_id, viewed_at, view_id;
