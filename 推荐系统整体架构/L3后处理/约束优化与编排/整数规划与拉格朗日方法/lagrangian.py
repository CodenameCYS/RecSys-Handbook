from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: str
    utility: float
    input_index: int


@dataclass(frozen=True)
class DualResult:
    selected_ids: tuple[str, ...]
    multipliers: dict[str, float]


def optimize_soft_caps(
    candidates: list[Candidate],
    target_counts: dict[str, int],
    top_k: int,
    iterations: int = 8,
    initial_step_size: float = 0.25,
) -> DualResult:
    """Apply subgradient updates for soft group upper targets.

    This routine does not enforce hard feasibility constraints. Those belong in
    the primary solver or an independent validator.
    """
    multipliers = {group: 0.0 for group in target_counts}

    for iteration in range(iterations):
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.utility - multipliers.get(candidate.group, 0.0)),
                candidate.input_index,
                candidate.candidate_id,
            ),
        )
        selected = ranked[:top_k]
        group_counts: dict[str, int] = {}
        for candidate in selected:
            group_counts[candidate.group] = group_counts.get(candidate.group, 0) + 1

        step_size = initial_step_size / sqrt(iteration + 1)
        for group, target in target_counts.items():
            violation = group_counts.get(group, 0) - target
            multipliers[group] = max(0.0, multipliers[group] + step_size * violation)

    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -(candidate.utility - multipliers.get(candidate.group, 0.0)),
            candidate.input_index,
            candidate.candidate_id,
        ),
    )
    return DualResult(
        selected_ids=tuple(candidate.candidate_id for candidate in ranked[:top_k]),
        multipliers={group: round(value, 4) for group, value in multipliers.items()},
    )
