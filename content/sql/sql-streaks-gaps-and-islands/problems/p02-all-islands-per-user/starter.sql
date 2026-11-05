-- visits(user_id BIGINT, visit_date DATE): one row per open; several rows can share a user and a date.
-- Return every streak of consecutive calendar days per user, one row per maximal run.
-- Output columns: user_id BIGINT, island_start DATE, island_end DATE, island_days BIGINT.
-- Duplicate visits on one day count once. Order by user_id, then island_start.
SELECT 1;
