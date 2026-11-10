-- Correct solution: position and balance come from correlated subqueries instead of window functions.
WITH ordered AS (
    SELECT
        p.account_id,
        p.posting_id,
        p.posted_at,
        p.amount_cents,
        (
            SELECT COUNT(*)
            FROM postings q
            WHERE q.account_id = p.account_id
              AND (q.posted_at, q.posting_id) <= (p.posted_at, p.posting_id)
        ) AS position_in_account,
        (
            SELECT SUM(q.amount_cents)
            FROM postings q
            WHERE q.account_id = p.account_id
              AND (q.posted_at, q.posting_id) <= (p.posted_at, p.posting_id)
        ) AS running_sum
    FROM postings p
)
SELECT
    account_id,
    posting_id,
    posted_at,
    amount_cents,
    CAST(position_in_account AS BIGINT) AS position_in_account,
    CAST(running_sum AS BIGINT) AS balance_cents
FROM ordered
ORDER BY account_id ASC, position_in_account ASC;
