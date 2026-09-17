from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    item_id: str
    entity_key: str
    score: float
    quality: float
    freshness: int
    input_index: int


@dataclass(frozen=True)
class Suppression:
    item_id: str
    entity_key: str
    kept_item_id: str
    reason_code: str = "duplicate_entity"


def deduplicate_by_entity(
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[Suppression]]:
    """Keep one deterministic representative for every entity key."""
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            -candidate.quality,
            -candidate.freshness,
            candidate.input_index,
            candidate.item_id,
        ),
    )
    kept_by_entity: dict[str, Candidate] = {}
    kept: list[Candidate] = []
    suppressed: list[Suppression] = []

    for candidate in ordered:
        representative = kept_by_entity.get(candidate.entity_key)
        if representative is None:
            kept_by_entity[candidate.entity_key] = candidate
            kept.append(candidate)
            continue
        suppressed.append(
            Suppression(
                item_id=candidate.item_id,
                entity_key=candidate.entity_key,
                kept_item_id=representative.item_id,
            )
        )

    return kept, suppressed