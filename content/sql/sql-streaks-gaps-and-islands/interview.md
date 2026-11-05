# Interview notes — SQL streaks (gaps and islands)

## Q1. How do you find each user's longest streak of consecutive active days?
**60-second answer.** Remove duplicate user-day pairs first, so a day counts once. Then give each run a constant key:
subtract the user's row number from the date, which is constant while the days stay consecutive. Group by user and key,
take the min and max date and the count, and rank by length. On a tie I take the earlier start, and I say so in the query.
**Follow-ups.** Why do duplicates break it? A repeated day advances the row number but not the date, so the key changes
inside the run. How would I do the same with LAG? Flag each day where the gap to the previous day is not one, take a running
sum of the flags per user, and group by that sum.

## Q2. How would you change the rule to "at most two days apart"?
**60-second answer.** Only the flag changes: the LAG form marks a new island when the gap is greater than two. The
row-number key does not handle that case, because a gap of two moves the date by two and the row number by one, so the
key shifts and the run splits. That is why the LAG form is easier to adapt.
**Follow-ups.** What about timezones? Convert timestamps to the user's local date before the distinct step, or a visit
near midnight can create or erase a streak.

## Common mistakes
- Skipping DISTINCT, so duplicates split islands.
- Grouping only by the key, which merges islands across users.
- Leaving the tie rule implicit.
