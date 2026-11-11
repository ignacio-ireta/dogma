-- Wrong on purpose: "previous" is the row for the calendar day before, not the previous recorded row.
SELECT
    cur.account_id,
    cur.usage_date,
    cur.api_calls,
    prev.usage_date AS previous_usage_date,
    prev.api_calls AS previous_api_calls,
    cur.api_calls - prev.api_calls AS calls_change
FROM account_daily_usage AS cur
LEFT JOIN account_daily_usage AS prev
    ON prev.account_id = cur.account_id AND prev.usage_date = cur.usage_date - 1
ORDER BY cur.account_id, cur.usage_date;
