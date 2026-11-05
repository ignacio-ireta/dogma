-- postings(account_id BIGINT, posting_id BIGINT, posted_at TIMESTAMP, amount_cents BIGINT)
-- Return each posting's running balance within its account, with a unique order per account:
--   (account_id, position_in_account, balance_cents) where balance_cents is the running sum
--   over (posted_at, posting_id), so every posting in a tie gets its own balance.
-- Output columns: account_id, posting_id, posted_at, amount_cents, position_in_account,
--   balance_cents (all BIGINT except posted_at TIMESTAMP). Order by account_id, position.
SELECT 1;
