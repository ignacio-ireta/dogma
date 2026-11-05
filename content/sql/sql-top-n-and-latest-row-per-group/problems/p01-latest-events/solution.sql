WITH ranked AS (
    SELECT
        entity_id,
        event_id,
        occurred_at,
        payload,
        ROW_NUMBER() OVER (
            PARTITION BY entity_id
            ORDER BY occurred_at DESC NULLS LAST, event_id DESC
        ) AS rank_in_entity
    FROM events
    WHERE kind = 'success'
)
SELECT entity_id, event_id, occurred_at, payload, rank_in_entity
FROM ranked
WHERE rank_in_entity <= 2
ORDER BY entity_id, rank_in_entity;
