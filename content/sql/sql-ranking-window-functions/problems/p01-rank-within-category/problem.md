# Rank products within each category

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that ranks every product inside its
category by revenue, three ways at once: `ROW_NUMBER`, `RANK` and `DENSE_RANK`. Every product appears in
the output, and the output is ordered so that the result is always the same.

## Contract

- Partition by the exact category string. Case and whitespace differences make separate groups.
- Input: a table `products` with `product_id` (BIGINT, unique), `category` (VARCHAR) and `revenue`
  (DOUBLE, may be NULL). One row per product.
- Partition: `category`. Rankings never cross categories.
- Order inside a category: `revenue` descending. **NULL revenue ranks last**, after every number,
  including 0. Two NULL revenues are peers of each other.
- `row_number`: `ROW_NUMBER()` over the partition. Ties in revenue are broken by `product_id`
  ascending, so every row gets a different number.
- `rank`: `RANK()` over the partition, ordered by revenue only. Equal revenues share a rank, and the
  next rank skips (1, 1, 3).
- `dense_rank`: `DENSE_RANK()` over the partition, ordered by revenue only. Equal revenues share a rank,
  and the next rank does not skip (1, 1, 2).
- Returns one row per product, with the columns in this order: `category`, `product_id`, `revenue`,
  `row_number`, `rank`, `dense_rank`.
- Order of rows: `category` ascending, then `row_number` ascending.
- If `products` is empty, return zero rows with the same six columns.
- Must read from `products`. Do not write the answer as literal rows.

## Examples

On `fixtures/products.csv`, the first rows of the result are:

```text
category  product_id  revenue  row_number  rank  dense_rank
books     101         250.00   1           1     1
books     103         250.00   2           1     1
books     102         180.50   3           3     2
books     105         90.00    4           4     3
books     104         NULL     5           5     4
books     106         NULL     6           5     4
```

Two products tie at 250.00: `ROW_NUMBER` gives 101 the number 1 and 103 the number 2 (the smaller
`product_id` first), while `RANK` and `DENSE_RANK` give both 1. The two NULL revenues come last, and they
share rank 5 (`RANK`) and rank 4 (`DENSE_RANK`).

```text
category  product_id  revenue    row_number  rank  dense_rank
tools     308         15.00      1           1     1
tools     905113208   15.00      2           1     1
tools     306         NULL       3           3     2
tools     307         NULL       4           3     2
```

A `product_id` such as `905113208` is compared as a number, not as text.

The event_log dataset can supply product ids, dirty category labels, and lognormal amounts. Its
category spellings stay separate ranking groups.

## Constraints

- Allowed: `SELECT`, `FROM`, `WHERE`, `WINDOW`, `ORDER BY`, common table expressions (`WITH`), and
  DuckDB's window functions.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `products` table.
The families they cover: ties on revenue with the tie-break decided by `product_id`; ids whose text order
differs from their numeric order; NULL revenues after every number, including 0, and peers of each other;
a category where every revenue is NULL; a single-product category; large, non-sequential ids; a table
inserted in shuffled order, so an unordered result would show; an empty table; dirty category keys kept
as stored, lognormal amounts, and `event_log` rows checked against an independent pandas computation
with `groupby().rank()`.
