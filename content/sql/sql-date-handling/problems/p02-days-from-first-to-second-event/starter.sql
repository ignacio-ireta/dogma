-- Problem: calendar days from each user's first event to their second event.
--
-- Tables:
--   events
--     user_id   BIGINT
--     event_ts  TIMESTAMP   -- one row per event; naive, no time zone. Two events may share a timestamp.
--
-- For each user with at least two events, return one row:
--   user_id          the user
--   days_to_second   BIGINT: calendar days between the DATE of the first event and the DATE of the
--                    second event. The first and second events are the two earliest event_ts values
--                    of the user; a tie at the first timestamp makes the second event the same time.
-- Users with fewer than two events are not returned.
-- Order by user_id ascending.
--
-- Replace the line below with your query.

SELECT 1;
