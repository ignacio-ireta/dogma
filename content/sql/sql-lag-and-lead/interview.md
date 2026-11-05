# Interview notes — SQL LAG and LEAD

## Q1. For each account, what changed since the previous day?
**60-second answer.** I name the partition (account), the order (date, unique per account), and the
default (NULL for the first day). Then `LAG(calls)` over that window gives the previous value, and
the change is current minus previous. I return the previous date too, so the comparison can be
checked. The result keeps every row and the output is ordered by account and date.
**Follow-ups.** What if a day is missing? The predecessor is the previous recorded row, not the
previous calendar day; if the business wants the calendar gap, I would generate the dates first.
Why not coalesce to zero? Zero is a real value and would report a change that never happened.

## Q2. How do you split events into sessions by a 30-minute inactivity gap?
**60-second answer.** Per visitor, ordered by time and then by a unique id, I take the previous
timestamp with `LAG`. A view starts a session if there is no previous view or the gap is strictly
greater than 30 minutes. A running `SUM` of those start flags, ordered the same way, numbers the
sessions. Exactly 30 minutes continues the session, which is the boundary a reviewer checks first.
**Follow-ups.** Why the unique id? Ties at a boundary decide which row carries the start flag, and
the output must not depend on the engine's order. Why a running sum rather than a self-join? The sum
is one pass over the ordered rows, and a self-join is much harder to keep correct at the boundary.

## Common mistakes
- Filtering with WHERE before LAG, which removes rows from the sequence.
- Ordering only by timestamp when timestamps can tie.
- Using COALESCE(LAG(x), 0) when the first row must be NULL.
- Comparing with `>=` instead of `>` at the threshold.
