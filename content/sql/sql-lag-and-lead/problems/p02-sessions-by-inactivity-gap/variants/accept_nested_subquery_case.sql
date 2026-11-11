-- Correct, with a nested subquery and a CASE instead of a CTE and an OR.
SELECT
    visitor_id,
    view_id,
    viewed_at,
    CAST(
        SUM(CAST(starts_session AS INTEGER)) OVER (
            PARTITION BY visitor_id
            ORDER BY viewed_at, view_id
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS BIGINT
    ) AS session_number,
    starts_session
FROM (
    SELECT
        visitor_id,
        view_id,
        viewed_at,
        CASE
            WHEN LAG(viewed_at) OVER w IS NULL THEN TRUE
            WHEN viewed_at - LAG(viewed_at) OVER w > INTERVAL 30 MINUTE THEN TRUE
            ELSE FALSE
        END AS starts_session
    FROM screen_views
    WINDOW w AS (PARTITION BY visitor_id ORDER BY viewed_at, view_id)
) AS flagged
ORDER BY visitor_id, viewed_at, view_id;
