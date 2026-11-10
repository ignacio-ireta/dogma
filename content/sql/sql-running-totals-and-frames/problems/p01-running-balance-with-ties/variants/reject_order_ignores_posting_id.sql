-- Wrong on purpose: the order inside each account uses posted_at only, so tied postings have no defined order.
WITH ordered AS (
    SELECT
        account_id,
        posting_id,
        posted_at,
        amount_cents,
        ROW_NUMBER() OVER (
            PARTITION BY account_id
            ORDER BY posted_at ASC
        ) AS position_in_account,
        SUM(amount_cents) OVER (
            PARTITION BY account_id
            ORDER BY posted_at ASC
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
