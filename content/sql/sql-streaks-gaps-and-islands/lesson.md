---
title: "SQL streaks (gaps and islands)"
minutes: 12
min_viable: "Read The system and the Worked example, then solve p01 with Hint 1 open."
---

## The system

A **streak** (island) is a maximal run of consecutive calendar days on which a user was active. Let
$d_1 < \dots < d_n$ be user $u$'s distinct active dates. Neighbours $d_k$ and $d_{k+1}$ share an island exactly when
$d_{k+1} - d_k = 1$ day. A gap of two or more days starts a new island.

Four rules decide almost every query:

1. **Remove duplicates first.** Use `SELECT DISTINCT user_id, visit_date` before any window function. A duplicate
   advances a row number but not the date, and the key below breaks.
2. **Give each island a key that is constant inside it.** The key moves with the date inside a run and changes at a gap.
3. **Group by (user, key).** The group gives the island's start, end and length.
4. **Decide the tie rule up front.** For the longest streak, equal lengths go to the earliest start.

Two ways to build the key:

| Method | Key | Use when |
|---|---|---|
| Date minus row number | $d_k - k$, with $k$ the row number within the user | you want one `GROUP BY` |
| LAG flag plus running SUM | $f_k = [d_k - d_{k-1} \ne 1]$, then $g_k = \sum_{j \le k} f_j$ | you want the gaps visible, or the rule is more than "exactly one day" |

## Worked example

User 1 visited on Jan 1, Jan 2, Jan 2 (again), Jan 3, Jan 5 and Jan 6. The distinct dates are Jan 1, 2, 3, 5, 6.

| $k$ | $d_k$ | $d_k - k$ | $f_k$ | $g_k$ |
|---|---|---|---|---|
| 1 | Jan 1 | Dec 31 | 1 (first) | 1 |
| 2 | Jan 2 | Dec 31 | 0 | 1 |
| 3 | Jan 3 | Dec 31 | 0 | 1 |
| 4 | Jan 5 | Jan 1 | 1 (gap of 2) | 2 |
| 5 | Jan 6 | Jan 1 | 0 | 2 |

Both methods give two islands: Jan 1 to 3 (length 3) and Jan 5 to 6 (length 2). The longest is the first.

**The duplicate trap.** Number the six raw visits instead. The keys become Dec 31, Dec 31, Dec 30, Dec 30, Dec 31,
Dec 31. Grouping by key gives one wrong island of four rows from Jan 1 to Jan 6, and a second from Jan 2 to Jan 3.
No error is raised.

## Why it works

Inside one island, each step adds one day to the date and one to the row number, so their difference is constant.
Across a gap of $g \ge 2$ days, the date moves by $g$ and the row number by $1$. The difference changes by $g - 1 \ge 1$.
The argument needs distinct dates: a duplicate moves the row number with no change in the date.

The LAG form makes the same argument visible. `LAG` returns the previous distinct day of the same user. A user's
first row gets `NULL`, so the `ELSE` branch sets its flag to 1. A running `SUM` of the flags, restarted per user,
numbers the islands. The frame is covered in [SQL running totals and frames](../sql-running-totals-and-frames/).

DuckDB date arithmetic: `DATE` minus an integer is a `DATE`, and `DATE` minus `DATE` is a count of days. It rejects
`DATE` minus `BIGINT`, which is what `ROW_NUMBER()` returns, so cast the row number to `INTEGER`.

## Signal vs. noise

Signal: distinct days first, the constant key, and grouping by (user, key). Noise: which key you pick, as long as the
output is right. Interviewers ask for both views, so know both.

## Where it breaks

- **Duplicates.** Without `DISTINCT`, the row-number key splits islands at repeated days. The result looks plausible.
- **Grouping only by the key.** Two users can share a key. Group by user too.
- **The tie rule.** Equal-length islands need an explicit order in the rank. Otherwise the answer can change between runs.
- **Unintended bridging.** "At most two days apart" is a different query. Put the threshold in the flag.
- **Time zones.** A timestamp converted to a date in the wrong zone moves visits across midnight. Convert before the distinct step.

A key that looks like a date number can pass the small examples and still fail on real logs. `year * 365 + day of year`, minus the row number, ignores that a leap year has 366 days, so it splits any run that crosses 31 December of a leap year. On a holiday burst of 40 users, 28 users get the wrong longest streak. User 1 has a 20-day streak from 20 December 2028 to 8 January 2029, and the key reports 12 days ending 31 December. Subtracting the row number from a real `DATE` avoids this, because the calendar does the counting.

## Interview angle

Sixty-second answer: a streak is a run of consecutive calendar days per user. Remove duplicate user-day pairs first.
Then either subtract the row number from the date, giving one key per run, or flag each day with `LAG` and take a running
sum. Group by user and key, take the min, max and count, and rank by length with the earliest start as the tie rule.
Likely follow-up: why does the duplicate step come first? A repeated day moves the row number but not the date, so the
key changes inside a run. Second follow-up: what changes for "at most two days apart"? Only the flag's threshold, which
is why the LAG form is easier to adapt.

## Connections

- Prerequisites: [SQL LAG and LEAD](../sql-lag-and-lead/) for the previous-row comparison, and
  [SQL running totals and frames](../sql-running-totals-and-frames/) for the running sum.
- Next: [SQL cohort retention](../sql-cohort-retention/), which applies the same date arithmetic to cohorts.
