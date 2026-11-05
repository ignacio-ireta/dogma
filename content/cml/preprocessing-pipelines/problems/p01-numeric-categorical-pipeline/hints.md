## Hint 1 — concept

Preprocessing is a set of rules estimated from one table and then replayed on another. Every number the
rules need, such as a median, a mode, a mean, a standard deviation or a list of categories, must come from
the training rows. The test rows only get looked up.

## Hint 2 — formula / approach

For each numeric column, compute the training median of observed values, fill the gaps in both tables
with it, then standardize both tables with the training mean and population std. For each categorical
column, compute the training mode with ties going to the smallest value, fill the gaps in the training
table, and take the sorted categories. Build a 0/1 block per categorical column, and let an unseen test
value fall through to an all-zero row.

## Hint 3 — code shape

Validate the requested columns first. Then loop over `numeric`, building one `(rows, 1)` block for each
column in each table, and loop over `categorical`, building one `(rows, k)` block for each column in each
table. Concatenate the blocks column-wise, separately for train and test, and return the two float64 arrays.
