-- Wrong on purpose: emits a NULL-filled row for each entity that has no eligible event (no-row rule C6)
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
),
entities AS (
    SELECT DISTINCT entity_id FROM events
)
SELECT
    e.entity_id,
    r.event_id,
    r.occurred_at,
    r.payload,
    CAST(r.rn AS BIGINT) AS rank_in_entity
FROM entities AS e
LEFT JOIN ranked AS r
    ON r.entity_id = e.entity_id AND r.rn <= 2
ORDER BY e.entity_id ASC, rank_in_entity ASC;
