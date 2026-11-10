-- Wrong on purpose: ranks every event first and filters to success afterwards, so a failure can take a top position (filter C1)
WITH ranked AS (
    SELECT
        entity_id,
        event_id,
        occurred_at,
        kind,
        payload,
        row_number() OVER (
            PARTITION BY entity_id
            ORDER BY occurred_at DESC NULLS LAST, event_id DESC
        ) AS rn
    FROM events
)
SELECT
    entity_id,
    event_id,
    occurred_at,
    payload,
    CAST(rn AS BIGINT) AS rank_in_entity
FROM ranked
WHERE rn <= 2 AND kind = 'success'
ORDER BY entity_id ASC, rank_in_entity ASC;
