-- Wrong on purpose: the change is previous minus current, so a rise shows as negative.
SELECT
    account_id,
    usage_date,
    api_calls,
    LAG(usage_date) OVER w AS previous_usage_date,
    LAG(api_calls) OVER w AS previous_api_calls,
    LAG(api_calls) OVER w - api_calls AS calls_change
FROM account_daily_usage
WINDOW w AS (PARTITION BY account_id ORDER BY usage_date)
ORDER BY account_id, usage_date;
