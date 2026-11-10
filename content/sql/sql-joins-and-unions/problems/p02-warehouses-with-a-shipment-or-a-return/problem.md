# Warehouses with a shipment or a return

Write one DuckDB SQL `SELECT` statement, in `submission.sql`, that returns every warehouse code
that handled at least one shipment or at least one return. Each code appears once, codes are
sorted, and a code that appears in both tables is still one row.

## Contract

- Input: two tables.
  - `shipments`: `shipment_id` (BIGINT), `warehouse_code` (VARCHAR, may be NULL), `shipped_on` (DATE).
  - `returns`: `return_id` (BIGINT), `warehouse_code` (VARCHAR, may be NULL), `returned_on` (DATE).
- Returns: one column, named `warehouse_code`, with one row per distinct non-NULL code that appears
  in `shipments`, in `returns`, or in both.
- Codes are compared and returned exactly as stored. `bos-1` and `BOS-1` are different codes.
  Do not change case or spacing.
- NULL codes never appear in the result.
- Row order: `warehouse_code` ascending. Row order is part of the answer.
- If neither table has a non-NULL code, return zero rows with the one column.
- Must read from `shipments` and `returns`. Do not write the answer as literal rows.

## Examples

On `fixtures/shipments.csv` and `fixtures/returns.csv`:

```text
warehouse_code
BOS-1
DEN-2
LAX-3
ORD-4
PHX-6
RNO-7
SEA-5
bos-1
```

`BOS-1` and `LAX-3` appear in both tables, and each is still one row. `RNO-7` and `PHX-6` appear
only in `returns`, and they are kept. The NULL codes in both tables do not appear. `bos-1` is
lowercase, so it is a different code and comes after every uppercase code.

A small case, with `shipments.warehouse_code` = `A, A, NULL` and `returns.warehouse_code` = `A, B`:

```text
warehouse_code
A
B
```

In the event_log dataset, category labels can include casing and spacing noise. If those labels
are warehouse codes, each stored spelling is still its own code.

## Constraints

- Allowed: standard `SELECT`, `FROM`, `WHERE`, `UNION` and `UNION ALL`, `ORDER BY`, and DuckDB's
  built-in operators.
- Not allowed: literal `VALUES` rows or any other hard-coded answer. Not allowed: normalising
  codes with `UPPER`, `LOWER` or `TRIM`.

## Done looks like

All public tests pass, then **Submit** runs the hidden tests. They build their own `shipments` and
`returns` tables. The hidden tests cover these families: codes that appear in only one table, codes
that appear in both, NULL codes, case and whitespace variants kept as separate keys, sorted output
with rows inserted out of order, one empty table, both tables empty, and dirty `event_log` categories
checked against a pandas union oracle.
