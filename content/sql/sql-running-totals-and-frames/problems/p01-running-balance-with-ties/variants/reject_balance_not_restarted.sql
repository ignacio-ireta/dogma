-- Wrong on purpose: the running sum has no PARTITION BY, so balances carry over from one account to the next.
WITH ordered AS (
    SELECT
        account_id,
        posting_id,
        posted_at,
        amount_cents,
        ROW_NUMBER() OVER (
            PARTITION BY account_id
            ORDER BY posted_at ASC, posting_id ASC
        ) AS position_in_account,
        SUM(amount_cents) OVER (
            ORDER BY account_id ASC, posted_at ASC, posting_id ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS running_sum
    FROM postings
)
SELECT
    account_id,
    posting_id,
    posted_at,
    amount_cents,
    position_in_account,
    CAST(running_sum AS BIGINT) AS balance_cents
FROM ordered
ORDER BY account_id ASC, position_in_account ASC;
