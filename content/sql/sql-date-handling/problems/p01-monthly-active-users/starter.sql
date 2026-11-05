-- Problem: active users for each calendar month in a stated range, including empty months.
--
-- Tables:
--   events
--     user_id   BIGINT
--     event_ts  TIMESTAMP   -- one row per event; naive local time, no time zone
--
-- Range: every calendar month from 2023-11 to 2024-02, inclusive (4 months).
-- Returns one row per month, ordered by month ascending, with these columns:
--   month          DATE: the first day of the month (not a TIMESTAMP)
--   active_users   BIGINT: distinct user_id with at least one event in that month, 0 if none
-- An event belongs to a month when it is at or after the month's first instant and before the
-- first instant of the next month. Events outside the range are ignored.
--
-- Replace the line below with your query.

SELECT 1;
