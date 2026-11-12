# Top two revenues per category, with ties

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns, for each category, every
product whose `DENSE_RANK` by revenue is 1 or 2. Ties at the cut are kept: if three products share the
second-highest revenue, all three come back, so a category can return more than two rows. A product
that is not in the top two dense ranks is not returned.

## Contract

- Categories are compared exactly as stored. Case and whitespace differences create separate groups.
- Input: a table `products` with `product_id` (BIGINT, unique), `category` (VARCHAR) and `revenue`
  (DOUBLE, may be NULL). One row per product.
- Dense rank inside a category: `DENSE_RANK()` ordered by `revenue` descending, with **NULL revenue
  ranked last** and NULLs treated as peers of each other. Equal revenues share a rank and the next rank
  does not skip. A category whose revenues are all NULL has every product at dense rank 1.
- Keep every product whose dense rank is at most 2. Do not keep a product only because of its
  `product_id`.
- Returns one row per kept product, with the columns in this order: `category`, `product_id`,
  `revenue`, `dense_rank`.
- Order of rows: `category` ascending, then `dense_rank` ascending, then `product_id` ascending.
- If no product qualifies, return zero rows with the same four columns.
- Must read from `products`. Do not write the answer as literal rows.

## Examples

On `fixtures/products.csv`, the first rows of the result are:

```text
category  product_id   revenue  dense_rank
audio     11           500.00   1
audio     12           500.00   1
audio     13           300.00   2
audio     14           300.00   2
audio     905113208    300.00   2
```

Products 13, 14 and 905113208 tie at 300.00, which is the second dense rank, so all three come back.
Product 15, at 120.00, is dense rank 3 and is not returned.

A second case, from `lenses` on the same fixture: products 41, 42 and 43 tie at 75.00 (dense rank 1), and product 44 is at 60.00 (dense rank 2), so all four come back:

```text
category  product_id  revenue  dense_rank
lenses    41          75.00    1
lenses    42          75.00    1
lenses    43          75.00    1
lenses    44          60.00    2
```

For `mounts`, every revenue is NULL. The three products are peers at dense rank 1, so all three come back.
For `video`, the two NULL products are dense rank 3 (after 80.00 and 10.00), so they are not returned.

Real rows (invoice lines of the `online_retail` data) supply ids, hand-typed category labels and
amounts that repeat inside a category. Rank each exact stored category independently before
keeping its first two dense ranks.

## Constraints

- Allowed: `SELECT`, `FROM`, `WHERE`, `ORDER BY`, subqueries, common table expressions (`WITH`), and
  DuckDB's window functions.
- Not allowed: literal `VALUES` rows or any other hard-coded answer.
- A window function cannot appear in `WHERE` directly. Compute the rank in a subquery or a CTE first.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `products` tables. The
families they cover: a tie at the cut that brings in more than two rows; a tie that straddles the second
rank; NULL revenues that rank below every number and are kept only when nothing else fills the top two;
an all-NULL category; a single-product category; large, non-sequential ids; a result with no row; a table
inserted in shuffled order; category keys kept as stored, repeated and negative amounts, and real retail rows
checked against an independent pandas computation with `groupby().rank(method="dense")`.

The hidden tests draw their bulk inputs from the real dataset `online_retail` (see
`docs/DATASETS.md`).
