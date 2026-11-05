-- visits(user_id BIGINT, visit_date DATE): one row per open; several rows can share a user and a date.
-- Return each user's longest streak of consecutive calendar days, with its first and last day.
-- Output columns: user_id BIGINT, streak_days BIGINT, streak_start DATE, streak_end DATE.
-- One row per user with at least one visit, ordered by user_id. On a tie in length, the earlier
-- streak_start wins. Duplicate visits on one day count once.
SELECT 1;
