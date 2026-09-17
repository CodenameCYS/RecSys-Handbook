from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    item_id: str
    group_id: str
    utility: float
    input_index: int


@dataclass(frozen=True)
class ScoredCandidate:
    item_id: str
    group_id: str
    utility: float
    adjustment: float
    adjusted_score: float


def calculate_adjustments(
    weighted_exposure: Mapping[str, float],
    target_share: Mapping[str, float],
    learning_rate: float,
    max_adjustment: float,
) -> dict[str, float]:
    """Calculate bounded soft adjustments from one version-consistent snapshot."""
    if learning_rate < 0 or max_adjustment < 0:
        raise ValueError("learning_rate and max_adjustment must be non-negative")

    active_groups = set(weighted_exposure) | set(target_share)
    if not active_groups:
        return {}

    total_exposure = sum(weighted_exposure.get(group_id, 0.0) for group_id in active_groups)
    if total_exposure < 0:
        raise ValueError("weighted_exposure must not contain negative totals")

    adjustments: dict[str, float] = {}
    for group_id in active_groups:
        actual_share = weighted_exposure.get(group_id, 0.0) / total_exposure if total_exposure else 0.0
        deficit = target_share.get(group_id, 0.0) - actual_share
        adjustments[group_id] = max(-max_adjustment, min(learning_rate * deficit, max_adjustment))
    return adjustments


def rerank_with_exposure_adjustment(
    candidates: Sequence[Candidate], adjustments: Mapping[str, float]
) -> list[ScoredCandidate]:
    """Rank already-eligible candidates by utility plus a bounded group adjustment."""
    scored = [
        ScoredCandidate(
            item_id=candidate.item_id,
            group_id=candidate.group_id,
            utility=candidate.utility,
            adjustment=adjustments.get(candidate.group_id, 0.0),
            adjusted_score=candidate.utility + adjustments.get(candidate.group_id, 0.0),
        )
        for candidate in candidates
    ]
    return sorted(scored, key=lambda candidate: (-candidate.adjusted_score, candidate.item_id))