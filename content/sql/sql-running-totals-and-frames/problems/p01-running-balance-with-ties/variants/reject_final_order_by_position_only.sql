-- Wrong on purpose: the final ORDER BY sorts by position only, so rows of different accounts interleave.
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
            PARTITION BY account_id
            ORDER BY posted_at ASC, posting_id ASC
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
ORDER BY position_in_account ASC;
