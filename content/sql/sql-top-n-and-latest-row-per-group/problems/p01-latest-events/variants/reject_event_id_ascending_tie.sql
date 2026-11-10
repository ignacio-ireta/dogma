-- Wrong on purpose: breaks timestamp ties by ascending event_id instead of descending (order C2)
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
        row_number() OVER (
            PARTITION BY entity_id
            ORDER BY occurred_at DESC NULLS LAST, event_id ASC
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
ORDER BY entity_id ASC, rank_in_entity ASC;
