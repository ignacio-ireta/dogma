-- account_daily_usage(account_id BIGINT, usage_date DATE, api_calls BIGINT)
-- One row per account per day with at least one call. Unique (account_id, usage_date).
-- Return one row per input row, ordered by account_id, usage_date, with:
--   previous_usage_date DATE, previous_api_calls BIGINT (NULL for the first day),
--   calls_change BIGINT = api_calls - previous_api_calls (NULL for the first day).
-- Use LAG, and do not fill in missing calendar days.
SELECT 1;
