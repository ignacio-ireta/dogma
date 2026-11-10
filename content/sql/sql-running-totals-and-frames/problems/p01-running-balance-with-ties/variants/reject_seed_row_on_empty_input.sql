-- Wrong on purpose: when postings is empty, a placeholder row is added so the output is never empty.
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
),
result AS (
    SELECT
        account_id,
        posting_id,
        posted_at,
        amount_cents,
        position_in_account,
        CAST(running_sum AS BIGINT) AS balance_cents
    FROM ordered
)
SELECT * FROM result
UNION ALL
SELECT 0, 0, TIMESTAMP '1970-01-01 00:00:00', 0, 0, 0
WHERE NOT EXISTS (SELECT 1 FROM postings)
ORDER BY account_id ASC, position_in_account ASC;
