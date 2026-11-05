---
title: SQL running totals and frames
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with an explicit ROWS frame."
---

## The system

A window aggregate such as `SUM(x) OVER (...)` computes one value per row. For each row $r$
it aggregates over a **frame**: the set of rows $F(r)$ in the same partition that the
function may see. Three clauses decide $F(r)$:

| Clause | Decides | Example |
|---|---|---|
| `PARTITION BY` | Which rows can ever be in the frame; each partition restarts the total | `PARTITION BY account_id` |
| `ORDER BY` | The position of each row, and which rows are **peers** (equal order values) | `ORDER BY posted_at, posting_id` |
| Frame: `ROWS`, `RANGE` or `GROUPS` | How the frame is measured from the current row | `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` |

The frame units measure differently:

- **ROWS** counts physical rows in the order. `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` is
  the current row and the six rows before it: seven rows when they exist.
- **RANGE** measures the ordering value. `RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW`
  is every row whose date $d_s$ satisfies $d_r - 6 \le d_s \le d_r$. A RANGE frame that ends at
  `CURRENT ROW` includes all peers of the current row, including peers later in the order.
- **GROUPS** counts peer groups. Rarely needed.

**The default frame.** With an `ORDER BY` and no frame clause, the frame is
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Without `ORDER BY`, it is the whole partition.

**Running-total rule.** Let $x_r$ be the amount on row $r$, and $\pi_r$ its position in a
**total** order (the order key plus a unique tiebreaker). The balance after row $r$ is

$$B_r = \sum_{s:\ \pi_s \le \pi_r} x_s.$$

A RANGE frame on the order key alone computes $\sum_{s:\ v_s \le v_r} x_s$, where $v$ is the
order value. The two agree only when no two rows share a value $v$.

## Worked example

One account has four postings. Posting 1 (+2), posting 2 (−1) and posting 3 (+5) all have
`posted_at` = 10:00. Posting 4 (+10) has `posted_at` = 11:00.

| posting_id | posted_at | amount | ROWS over (posted_at, posting_id) | RANGE over posted_at only (default frame) |
|---|---|---|---|---|
| 1 | 10:00 | +2 | 2 | 6 |
| 2 | 10:00 | −1 | 1 | 6 |
| 3 | 10:00 | +5 | 6 | 6 |
| 4 | 11:00 | +10 | 16 | 16 |

The balance after posting 2 is 1. The RANGE column reports 6 for all three tied postings,
because each frame includes peer 3, which is later in the order. Adding `posting_id` as the
tiebreaker gives the ROWS column: one distinct total per row.

For a trailing average, `ROWS BETWEEN 6 PRECEDING` takes seven rows regardless of dates. If a
shop has no sales on some days, seven rows can cover more than seven days.

## Why it works

A total order gives every row a distinct position, so the ROWS frame is the set of rows up to
that position. Its sum is a prefix sum (`algo/prefix-sums`), computed per partition. Peers break
the prefix property: a RANGE frame for the first tied row covers the whole tie group.

**Why write ROWS explicitly.** With a unique order, ROWS and the default RANGE give the same
totals, so ROWS is about clarity. It says "by position", which the default does not. It also limits
the damage of a later edit: if someone drops the tiebreaker, ROWS still gives each row its own
sum, while RANGE silently gives tied rows a shared total.

With a gap-free daily series, ROWS and RANGE agree on a trailing window. Gaps make them differ,
and only the RANGE form keeps the calendar meaning.

## Signal vs. noise

- **Signal:** the tiebreaker orders the rows; the frame defines each total. Make both visible.
- **Signal:** ask whether the window is the last $n$ rows or the last $n$ days, and whether the
  data has gaps. Those two questions decide ROWS or RANGE.
- **Noise for now:** GROUPS, `EXCLUDE` clauses and window-frame performance details.

## Where it breaks

- **Default frame with ties.** `SUM(x) OVER (PARTITION BY a ORDER BY t)` gives all tied rows the
  same total. It is the most common running-total bug.
- **No tiebreaker.** Tied rows take their positions in an arbitrary order, so the balances can
  change when the input order changes.
- **ROWS over calendar data with gaps.** The "last seven rows" can span more than seven days.
- **Integer sums in DuckDB.** `SUM` over `BIGINT` returns `HUGEINT`. Cast back to `BIGINT` when
  the schema requires it.
- **NULL order values.** Decide where they sort, with `NULLS FIRST` or `NULLS LAST`.

## Interview angle

State the partition (what resets the total), the total order (what makes positions unique) and
the frame, then write the OVER clause. Mention the default-frame trap first. For a moving average,
say whether the window is a row count or a calendar interval, and what happens on missing days.

## Connections

`ROW_NUMBER` from the ranking topic gives the total order that the frame needs. LAG and LEAD
reach a fixed offset inside the same ordered partition, a special case of a ROWS frame.
