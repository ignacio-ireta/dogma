# Interview notes: Beta topic

## Q1. How does GROUP BY work?

**60-second answer.** It splits rows into groups by key and reduces each group with an aggregate.
The result has one row per key.

**Follow-ups.**
- *NULL keys?* They form one group of their own.

## Q2. WHERE or HAVING?

**60-second answer.** WHERE filters rows before grouping. HAVING filters groups after aggregation.

## Common mistakes

- Selecting a column that is neither grouped nor aggregated.
