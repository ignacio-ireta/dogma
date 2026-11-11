-- Wrong on purpose: a missing predecessor is filled with 0 instead of NULL.
SELECT
    account_id,
    usage_date,
    api_calls,
    LAG(usage_date) OVER w AS previous_usage_date,
    COALESCE(LAG(api_calls) OVER w, 0) AS previous_api_calls,
    api_calls - COALESCE(LAG(api_calls) OVER w, 0) AS calls_change
FROM account_daily_usage
WINDOW w AS (PARTITION BY account_id ORDER BY usage_date)
ORDER BY account_id, usage_date;
