-- Correct, with a self-join instead of a window: each row finds its previous recorded day
-- as the latest earlier day of the same account.
SELECT
    cur.account_id,
    cur.usage_date,
    cur.api_calls,
    prev.usage_date AS previous_usage_date,
    prev.api_calls AS previous_api_calls,
    cur.api_calls - prev.api_calls AS calls_change
FROM account_daily_usage AS cur
LEFT JOIN account_daily_usage AS prev
    ON prev.account_id = cur.account_id
    AND prev.usage_date = (
        SELECT MAX(p.usage_date)
        FROM account_daily_usage AS p
        WHERE p.account_id = cur.account_id AND p.usage_date < cur.usage_date
    )
ORDER BY cur.account_id, cur.usage_date;
