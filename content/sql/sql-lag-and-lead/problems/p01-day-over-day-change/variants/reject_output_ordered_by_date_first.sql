-- Wrong on purpose: rows are ordered by usage date first, then by account.
SELECT
    account_id,
    usage_date,
    api_calls,
    LAG(usage_date) OVER w AS previous_usage_date,
    LAG(api_calls) OVER w AS previous_api_calls,
    api_calls - LAG(api_calls) OVER w AS calls_change
FROM account_daily_usage
WINDOW w AS (PARTITION BY account_id ORDER BY usage_date)
ORDER BY usage_date, account_id;
