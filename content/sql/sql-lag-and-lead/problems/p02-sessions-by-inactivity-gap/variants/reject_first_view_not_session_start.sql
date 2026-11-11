-- Wrong on purpose: a visitor first view does not open a session, so it is not counted.
-- each visitor has no predecessor, so previous_at is NULL.
WITH ordered AS (
    SELECT
        visitor_id,
        view_id,
        viewed_at,
        LAG(viewed_at) OVER w AS previous_at
    FROM screen_views
    WINDOW w AS (PARTITION BY visitor_id ORDER BY viewed_at, view_id)
),
-- Step 2: flag the views that open a session. The gap test is strict: exactly 30 minutes
-- continues the session.
flagged AS (
    SELECT
        visitor_id,
        view_id,
        viewed_at,
        COALESCE(viewed_at - previous_at > INTERVAL 30 MINUTE, FALSE) AS starts_session
    FROM ordered
)
-- Step 3: a running sum of the flags numbers the sessions. SUM over BOOLEAN-as-INTEGER returns
-- HUGEINT in DuckDB, so cast to BIGINT. ROWS makes the frame explicit: row by row, no peers.
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
FROM flagged
ORDER BY visitor_id, viewed_at, view_id;
