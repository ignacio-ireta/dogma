# Interview notes — SQL cohort retention

## Q1. Build a monthly retention table from an events log.
**60-second answer.** Define the cohort first: the month of each user's first event. Then the
period is the whole-month offset from that cohort month, and the cell counts the distinct users
of that cohort who were active in that period. The denominator is the cohort's full size, so each
row reads as a share of the cohort. I build four CTEs: first month per user, cohort sizes, distinct
active users per cohort and period, and the grid of every cohort and period, left-joined to the
counts so empty periods show as zero.
**Follow-ups.** Why not divide by the previous period? That turns the table into a chain of
conversion rates, and a user lost early is no longer in the base. Why a grid? Without it, a
period nobody returned in disappears, and the curve looks better than it is.

## Q2. A retention rate is above 1.0 for one cohort. What happened?
**60-second answer.** Almost always a count of events instead of users, or a denominator that
is not the cohort. I would check period 0 first: it should equal the cohort size, because every
user is active in their first month. If it is larger, the numerator counts rows. If it is
smaller, some users are attributed to the wrong month.
**Follow-ups.** How do you check a month boundary? Compare a user whose last event is 31 January
with one on 1 February. Should a user with activity in a later month but none in between be
counted? Yes, at the period where they are active; the gap is data, not an error.

## Q3. How would you handle weekly cohorts?
**60-second answer.** The same shape with a different truncation: truncate to the week start,
measure the offset in weeks, and keep the same denominator. The grid length changes too, since
the maximum period is the number of weeks between the cohort week and the last week in the data.
**Follow-ups.** What about ISO weeks across a year boundary? Use one truncation for the whole
query and say which one. Is a rolling window different? Yes, it asks whether a user was active in
a window, and that needs a separate definition stated up front.

## Common mistakes
- Counting events with COUNT(*) instead of COUNT(DISTINCT user_id).
- Dividing by the previous period's active users instead of the cohort size.
- Taking the cohort from the full timestamp, so each user's cohort is their first second.
- An inner join that drops zero cells, so the grid has holes.
- Dividing a day difference by 30 to get months.
