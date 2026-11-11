-- Blind solution, written from problem.md alone: rank each user's events both ways.
WITH ranked AS (
    SELECT
        user_id,
        event_time,
        event_type,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time ASC, event_id ASC) AS rn_first,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time DESC, event_id DESC) AS rn_last
    FROM events
)
SELECT
    f.user_id,
    f.event_time AS first_event_time,
    f.event_type AS first_event_type,
    l.event_time AS last_event_time,
    l.event_type AS last_event_type
FROM ranked AS f
JOIN ranked AS l ON l.user_id = f.user_id
WHERE f.rn_first = 1 AND l.rn_last = 1
ORDER BY f.user_id;
