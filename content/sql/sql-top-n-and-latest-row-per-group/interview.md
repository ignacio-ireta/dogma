# Interview notes — SQL top-N and latest row per group

## Q1. How would you select the latest record for each entity?
**60-second answer.** Define eligibility, event-time semantics, unknown-time policy
and a deterministic tie-breaker. Rank eligible records with ROW_NUMBER partitioned
by entity and ordered by timestamp descending plus a unique identifier. Keep
position one and return the columns from that selected row.
**Follow-ups.** Why not MAX per column? The maxima may come from different rows.
What if all tied latest records are required? Change the tie policy, usually ranking
by timestamp without the identifier and preserving peers.

## Q2. How does top-N differ from N distinct scores?
**60-second answer.** Top-N rows needs unique row positions. N distinct scores is
a dense-rank threshold. Including boundary peers can return more than N rows.
Specify the business rule first; then choose the ranking function and complete
ordering key. Add a final ORDER BY if callers need stable output order.
**Follow-ups.** Why filter first? Ineligible rows must not consume positions.
Does LIMIT N solve the grouped task? No; it limits the entire result.

## Common mistakes
- Ranking before filtering eligible records.
- Using ingestion identifiers as a substitute for event time.
- Returning unrelated maxima instead of one coherent row.
- Letting NULL placement or timestamp ties depend on implicit defaults.
