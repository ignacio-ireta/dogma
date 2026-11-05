-- Step 1 (the total order): ROW_NUMBER over (posted_at, posting_id) gives each posting a unique
-- position inside its account. The posting identifier breaks ties; without it, postings that
-- share a timestamp have no defined order.
--
-- Step 2 (the running sum): SUM over the same order with an explicit ROWS frame. The frame sums
-- the rows from the first position up to the current one. Because posting_id makes the order
-- unique, the default RANGE frame gives the same totals; ROWS states the intent. Without
-- posting_id, RANGE would give every tied posting the balance of the whole tie group.
--
-- Step 3: cast back to BIGINT. DuckDB widens SUM over BIGINT, and the schema promises BIGINT.
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
ORDER BY account_id ASC, position_in_account ASC;
