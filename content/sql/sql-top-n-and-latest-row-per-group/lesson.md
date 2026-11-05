---
title: SQL top-N and latest row per group
minutes: 12
min_viable: "Read The system and Worked example, then solve p01 with a filtered ranking CTE."
---

## The system

A top-N query chooses **rows**, not unrelated maxima from each column. Define four
things before writing SQL: eligible rows, grouping key, ordering key, and tie policy.
For event data, a larger event identifier is not necessarily a later timestamp.

The reliable pattern is to filter eligible rows, assign a position within each
group, then keep positions up to N. For one latest row, use N equal to one.
In p01, N is two, the group is entity_id, and eligible events have kind exactly
'success'. Order known timestamps descending, unknown timestamps last, and use
event_id descending to make every position deterministic.

| Policy | Ranking choice | Consequence |
|---|---|---|
| At most N rows per group | ROW_NUMBER | Each row receives a distinct position |
| Include all peers at a boundary | RANK | Ties share a position; later positions can have gaps |
| Keep N distinct ordering values | DENSE_RANK | Ties share a position without gaps |

Peers are defined by the complete window ordering key. Adding a unique identifier
removes ties in that key. Choose tie behavior deliberately; the function name alone
does not describe the full policy.

## Worked example

Entity 1 has these events:

| id | occurred_at | kind | payload |
|---|---|---|---|
| 1 | 1 January | success | old |
| 2 | 3 January | failure | ignored |
| 3 | 2 January | success | new |
| 4 | unknown | success | unknown |

Filter first: event 2 is ineligible. Ordering the survivors gives event 3 at position
one, event 1 at position two, and event 4 at position three. Keep events 3 and 1.
Ranking before filtering would let the failed event consume one of the two slots.

For entity 2, events 8 and 9 both occurred on 2 January. The explicit descending
identifier rule puts event 9 first. Its payload may be NULL; it must remain NULL.
The timestamp and payload returned for a row must come from that same row.

## Why it works

PARTITION BY makes each entity an independent ordered sequence. ROW_NUMBER labels
the sequence without collapsing rows. An outer query can then select complete
records by position. A CTE expresses the stages visibly; DuckDB also supports
QUALIFY for filtering window results. The
[DuckDB window reference](https://duckdb.org/docs/lts/sql/functions/window_functions)
and [QUALIFY documentation](https://www.duckdb.org/docs/current/sql/query_syntax/qualify)
describe these evaluation stages.

This is different from GROUP BY. Aggregating MAX(timestamp) identifies a value,
but not a unique complete record when timestamps tie. Joining that value back to
the table can produce multiple rows. Aggregating MAX(timestamp) and MAX(payload)
separately can even construct a pair that never appeared together.

## Signal vs. noise

- **Signal:** ask whether N counts rows or distinct values, whether boundary ties
  stay, and how unknown timestamps should behave.
- **Signal:** the final result needs its own ORDER BY. A window's ordering defines
  positions inside groups; it does not promise the order of returned rows.
- **Noise for now:** dialect shortcuts before the row-selection policy is clear.

## Where it breaks

A missing tie-breaker makes the chosen latest row unstable when timestamps tie.
Using event_id alone confuses ingestion order with event time. Applying LIMIT N
globally keeps N rows for the entire dataset rather than N per entity.

Unknown timestamps need a stated rule. They do not mean the Unix epoch, and an
arbitrary replacement can change ordering. Here they follow known timestamps.
String comparison is also insufficient for inconsistent timestamp formats; p01
supplies a typed TIMESTAMP column instead of asking the query to guess a format.

Window execution may sort and buffer substantial data. For a large event store,
filter irrelevant events early and examine the actual query plan. An appropriate
index can help some engines, but the physical benefit depends on the engine and
workload; correctness does not depend on that optimization.

## Interview angle

Start with the policy questions, then describe filter, partition, order, rank and
select. Explain the tie-breaker and how you preserve a coherent row. A strong
follow-up answer distinguishes the ordering that assigns positions from the
ordering of the final output.

## Connections

Week-6 ranking functions supply the mechanism. This topic turns them into latest
state, per-group shortlists and deduplication components. LAG/LEAD will next relate
a selected row to its neighbors in an ordered event sequence.
