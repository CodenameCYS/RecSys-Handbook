from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: str
    utility: float
    input_index: int


@dataclass(frozen=True)
class Quota:
    group: str
    minimum: int = 0
    maximum: int | None = None


def greedy_quota_select(
    candidates: list[Candidate], quotas: list[Quota], top_k: int
) -> list[Candidate]:
    """Select a feasible prefix for mutually exclusive, static groups."""
    quota_by_group = {quota.group: quota for quota in quotas}
    if len(quota_by_group) != len(quotas):
        raise ValueError("quota groups must be unique")

    for quota in quotas:
        if quota.minimum < 0 or (
            quota.maximum is not None and quota.maximum < quota.minimum
        ):
            raise ValueError(f"invalid quota for {quota.group}")

    required_slots = sum(quota.minimum for quota in quotas)
    if required_slots > top_k:
        raise ValueError("lower quotas exceed the requested list length")

    supply_by_group: dict[str, int] = {}
    for candidate in candidates:
        supply_by_group[candidate.group] = supply_by_group.get(candidate.group, 0) + 1

    for quota in quotas:
        if supply_by_group.get(quota.group, 0) < quota.minimum:
            raise ValueError(f"lower quota is infeasible for {quota.group}")

    remaining = sorted(
        candidates,
        key=lambda candidate: (-candidate.utility, candidate.input_index, candidate.candidate_id),
    )
    counts = {quota.group: 0 for quota in quotas}
    selected: list[Candidate] = []

    while remaining and len(selected) < top_k:
        remaining_slots = top_k - len(selected)
        required_remaining = sum(
            max(0, quota.minimum - counts[quota.group]) for quota in quotas
        )
        reserve_for_lower_quotas = remaining_slots == required_remaining

        chosen_index: int | None = None
        for index, candidate in enumerate(remaining):
            quota = quota_by_group.get(candidate.group)
            count = counts.get(candidate.group, 0)
            if quota is not None and quota.maximum is not None and count >= quota.maximum:
                continue
            if reserve_for_lower_quotas and (
                quota is None or count >= quota.minimum
            ):
                continue
            chosen_index = index
            break

        if chosen_index is None:
            break

        chosen = remaining.pop(chosen_index)
        selected.append(chosen)
        if chosen.group in counts:
            counts[chosen.group] += 1

    return selected
