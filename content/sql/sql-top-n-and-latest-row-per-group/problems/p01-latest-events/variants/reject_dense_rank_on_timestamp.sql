-- Wrong on purpose: ranks by timestamp alone, so tied timestamps share a position and return more than two rows (C3)
WITH eligible AS (
    SELECT entity_id, event_id, occurred_at, payload
    FROM events
    WHERE kind = 'success'
),
ranked AS (
    SELECT
        entity_id,
        event_id,
        occurred_at,
        payload,
        dense_rank() OVER (
            PARTITION BY entity_id
            ORDER BY occurred_at DESC NULLS LAST
        ) AS rn
    FROM eligible
)
SELECT
    entity_id,
    event_id,
    occurred_at,
    payload,
    CAST(rn AS BIGINT) AS rank_in_entity
FROM ranked
WHERE rn <= 2
ORDER BY entity_id ASC, rank_in_entity ASC, event_id DESC;
