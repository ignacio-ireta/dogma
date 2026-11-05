# Latest successful events per entity

Return at most the latest two successful events for each entity from the
`events` table. Preserve the complete selected records.

## Contract

Table `events` has columns:

| Column | DuckDB type | Rules |
|---|---|---|
| entity_id | BIGINT | Non-null grouping key |
| event_id | BIGINT | Non-null, globally unique identifier |
| occurred_at | TIMESTAMP | May be NULL; timestamps have no timezone conversion |
| kind | VARCHAR | Non-null; eligible only when exactly 'success' |
| payload | VARCHAR | May be NULL |

- First filter to eligible events.
- Within each entity order by `occurred_at DESC NULLS LAST, event_id DESC`.
- Return positions 1 and 2, or fewer if that entity has fewer eligible events.
- Output columns, in order: `entity_id BIGINT`, `event_id BIGINT`,
  `occurred_at TIMESTAMP`, `payload VARCHAR`, `rank_in_entity BIGINT`.
- Final output order is `entity_id ASC, rank_in_entity ASC`. Positions start at 1.
- Return no row for an entity with no eligible events. An empty table returns an
  empty result with the stated column names and types.
- Do not change the input table.

## Examples

```text
entity_id event_id occurred_at kind     payload
1         1        2026-01-01  success  old
1         2        2026-01-03  failure  ignored
1         3        2026-01-02  success  new
=> (1, 3, 2026-01-02, new, 1), (1, 1, 2026-01-01, old, 2)
```

```text
entity_id event_id occurred_at kind     payload
2         8        2026-01-02  success  eight
2         9        2026-01-02  success  NULL
2         10       NULL        success  unknown
=> (2, 9, 2026-01-02, NULL, 1), (2, 8, 2026-01-02, eight, 2)
```

## Constraints

Use a read-only DuckDB SQL query. CTEs, subqueries and QUALIFY are allowed.
The query is graded by its results, not by requiring a particular syntax.
Timestamps can tie and identifiers need not increase with time.

## Done looks like

The result matches an independent group-and-sort oracle on seeded records and
fixtures. Hidden cases cover ties, NULL timestamps and payloads, filtering order,
entities with no successes, empty input, exact BIGINT types, large identifiers
and final ordering. All values from a selected row remain together.
