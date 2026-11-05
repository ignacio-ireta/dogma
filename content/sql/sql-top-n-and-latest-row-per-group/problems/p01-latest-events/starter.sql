-- events(entity_id BIGINT, event_id BIGINT, occurred_at TIMESTAMP,
--        kind VARCHAR, payload VARCHAR)
-- Return the latest two 'success' events per entity, with rank_in_entity.
-- Timestamp descending NULLS LAST, then event_id descending; final order by entity and rank.
SELECT 1;
