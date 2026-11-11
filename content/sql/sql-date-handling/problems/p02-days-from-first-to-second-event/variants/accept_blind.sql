-- Blind: LEAD over the user's time order, keeping only the first event of each user.
WITH ordered AS (
    SELECT
        user_id,
        event_ts,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts) AS position,
        LEAD(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS second_ts
    FROM events
)
SELECT
    user_id,
    CAST(CAST(second_ts AS DATE) - CAST(event_ts AS DATE) AS BIGINT) AS days_to_second
FROM ordered
WHERE position = 1 AND second_ts IS NOT NULL
ORDER BY user_id;
