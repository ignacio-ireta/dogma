SELECT
    account_id,
    posting_id,
    posted_at,
    amount_cents,
    CAST(ROW_NUMBER() OVER w AS BIGINT) AS position_in_account,
    CAST(SUM(amount_cents) OVER w AS BIGINT) AS balance_cents
FROM postings
WINDOW w AS (
    PARTITION BY account_id
    ORDER BY posted_at ASC, posting_id ASC
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
)
ORDER BY account_id ASC, position_in_account ASC;
