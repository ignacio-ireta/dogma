-- Wrong on purpose: the look-back runs across all accounts, not inside each account.
SELECT
    account_id,
    usage_date,
    api_calls,
    LAG(usage_date) OVER w AS previous_usage_date,
    LAG(api_calls) OVER w AS previous_api_calls,
    api_calls - LAG(api_calls) OVER w AS calls_change
FROM account_daily_usage
WINDOW w AS (ORDER BY account_id, usage_date)
ORDER BY account_id, usage_date;
