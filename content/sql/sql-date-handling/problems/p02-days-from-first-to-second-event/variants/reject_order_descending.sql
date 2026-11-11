-- Reference query: calendar days from each user's first event to their second event.
--
-- Grain first: the answer has one row per user, and only users with at least two events have one.
-- Work from events ordered by time within each user: ROW_NUMBER picks position 1 (the first event) and
-- position 2 (the second event). Two events with the same timestamp are two events, so a tie gives 0.
--
-- The distance is a DATE difference: CAST each timestamp to DATE, then subtract. That counts calendar
-- days. Subtracting two TIMESTAMPs gives an INTERVAL, and date_part('day', interval) reads only the
-- day field of that interval, so 23:59 to 00:01 the next day would give 0 instead of 1.
WITH ranked AS (
    SELECT
        user_id,
        event_ts,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_ts) AS position
    FROM events
),
firsts AS (
    SELECT user_id, event_ts FROM ranked WHERE position = 1
),
seconds AS (
    SELECT user_id, event_ts FROM ranked WHERE position = 2
)
SELECT
    f.user_id,
    CAST(s.event_ts AS DATE) - CAST(f.event_ts AS DATE) AS days_to_second
FROM firsts AS f
INNER JOIN seconds AS s
    ON s.user_id = f.user_id   -- an inner join keeps only users with a second event
ORDER BY f.user_id DESC;
