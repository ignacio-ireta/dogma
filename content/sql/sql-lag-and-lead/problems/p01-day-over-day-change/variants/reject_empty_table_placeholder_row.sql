-- Wrong on purpose: an empty usage table returns one placeholder row of NULLs.
WITH base AS (
    SELECT
        account_id,
        usage_date,
        api_calls,
        LAG(usage_date) OVER w AS previous_usage_date,
        LAG(api_calls) OVER w AS previous_api_calls,
        api_calls - LAG(api_calls) OVER w AS calls_change
    FROM account_daily_usage
    WINDOW w AS (PARTITION BY account_id ORDER BY usage_date)
)
SELECT * FROM base
UNION ALL
SELECT
    NULL::BIGINT, NULL::DATE, NULL::BIGINT, NULL::DATE, NULL::BIGINT, NULL::BIGINT
WHERE NOT EXISTS (SELECT 1 FROM account_daily_usage)
ORDER BY 1, 2;
