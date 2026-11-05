## Hint 1 — concept

Choose the eligible rows before assigning positions. Keep the full selected row.

## Hint 2 — formula / approach

Partition by entity, order by the explicit timestamp and identifier rules, and
keep the first two unique positions in each partition.

## Hint 3 — code shape

A CTE can produce a ROW_NUMBER column after the success filter. The outer query
filters positions and orders the final output. State NULLS LAST explicitly.
