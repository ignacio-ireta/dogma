## Hint 1 — concept

The question defines the window by dates, not by how many rows there are. Ask what a window of
seven rows means for a shop that was closed on two of the last seven days.

## Hint 2 — formula / approach

Each row needs the rows of its own store whose date is at most six days before its date and not
after it. Put the store in the partition, order by the date, and choose a frame whose bounds are
measured in dates. Count the rows in the same frame for the second output column.

## Hint 3 — code shape

Use `AVG` and `COUNT` with the same `OVER` clause. In DuckDB a RANGE frame accepts an interval
offset on a date ordering, so write the offset as `INTERVAL 6 DAYS PRECEDING` in the frame.
