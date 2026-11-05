# Running balance per account, with tied posting times

A small card-terminal business records every posting to its customer accounts. Several
postings often share the same `posted_at` second, because a batch import stamps them together.
Write one query that returns the running balance of each account after each posting. Two postings
at the same time must get different balances, and each balance must include exactly the postings
that come before it.

## Contract

Table `postings` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| account_id | BIGINT | Non-null grouping key |
| posting_id | BIGINT | Non-null and unique across the table |
| posted_at | TIMESTAMP | Non-null; many postings can share one value |
| amount_cents | BIGINT | Non-null; negative for a debit, positive for a credit |

- Postings of one account are ordered by `posted_at ASC`, then `posting_id ASC`. The second key
  breaks ties, and the query must use it.
- `position_in_account` is 1 for the first posting in that order, 2 for the next, and so on.
- `balance_cents` is the sum of `amount_cents` over the postings of that account at positions up
  to and including this one. It restarts for every account.
- Output columns, in order, with these types: `account_id BIGINT`, `posting_id BIGINT`,
  `posted_at TIMESTAMP`, `amount_cents BIGINT`, `position_in_account BIGINT`,
  `balance_cents BIGINT`. DuckDB widens `SUM` over `BIGINT`, so cast the balance back to `BIGINT`.
- Final output order is `account_id ASC, position_in_account ASC`.
- The frame is your choice. Any query that returns the balances above is accepted, whatever
  frame keyword it uses. The grader checks results, not syntax.
- An empty table returns no rows, with the stated column names and types.
- Do not change the input table.

## Examples

```text
account_id posting_id posted_at            amount_cents
7          12         2026-03-01 09:00:00   500
7          10         2026-03-01 09:00:00  -120
7          11         2026-02-28 17:00:00    80
7          13         2026-03-02 08:00:00  -200
=> (7, 11, 2026-02-28 17:00:00,    80, 1,  80)
   (7, 10, 2026-03-01 09:00:00,  -120, 2, -40)
   (7, 12, 2026-03-01 09:00:00,   500, 3, 460)
   (7, 13, 2026-03-02 08:00:00,  -200, 4, 260)
```

Posting 10 ends at -40. If the order used only `posted_at`, the running sum would include the
tied posting 12 as well, and posting 10 would show 460. The posting identifier is what gives
each tied posting its own position.

```text
account_id posting_id posted_at            amount_cents
9          4          2026-01-05 00:00:00    75
=> (9, 4, 2026-01-05 00:00:00, 75, 1, 75)
```

## Constraints

- Allowed: a read-only DuckDB SQL query. CTEs, subqueries, window functions and `QUALIFY` are
  allowed.
- Not allowed: writing to or altering the `postings` table, and any approach that does not
  return one row per posting.

## Done looks like

The result matches an independent group-and-sort oracle on seeded postings with many tied times,
and on a fixture. Hidden cases cover tie groups of three or more, postings whose insertion order
is the reverse of their identifier order, balances that restart per account, single-posting
accounts, empty input, debits that make the balance negative, values beyond 2^53, exact output
types and final ordering. A query whose order ignores `posting_id` fails the tie cases, whatever
its frame.
