-- LAG looks back one row inside each account's partition, in date order. The first row of a
-- partition has no predecessor, so LAG returns its default, NULL. Nothing is coalesced to 0.
SELECT
    account_id,
    usage_date,
    api_calls,
    LAG(usage_date) OVER w AS previous_usage_date,
    LAG(api_calls) OVER w AS previous_api_calls,
    api_calls - LAG(api_calls) OVER w AS calls_change
FROM account_daily_usage
WINDOW w AS (PARTITION BY account_id ORDER BY usage_date)
ORDER BY account_id, usage_date;
