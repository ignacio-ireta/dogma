-- Wrong on purpose: final output sorted by rank before entity, not entity then rank (order C5)
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
            ORDER BY occurred_at DESC NULLS LAST, event_id DESC
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
ORDER BY rank_in_entity ASC, entity_id ASC;
