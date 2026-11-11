-- Correct solution: first and last picked by a window with QUALIFY, then joined back per user.
WITH firsts AS (
    SELECT user_id, event_time AS first_event_time, event_type AS first_event_type
    FROM events
    QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY event_time ASC, event_id ASC) = 1
),
lasts AS (
    SELECT user_id, event_time AS last_event_time, event_type AS last_event_type
    FROM events
    QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY event_time DESC, event_id DESC) = 1
)
SELECT f.user_id, f.first_event_time, f.first_event_type, l.last_event_time, l.last_event_type
FROM firsts AS f
JOIN lasts AS l USING (user_id)
ORDER BY f.user_id;
