from __future__ import annotations

from dataclasses import dataclass

try:
    from ortools.sat.python import cp_model
except ModuleNotFoundError:
    cp_model = None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    group: str
    utility: float
    allowed_positions: frozenset[int]
    input_index: int


@dataclass(frozen=True)
class Quota:
    group: str
    minimum: int = 0
    maximum: int | None = None


@dataclass(frozen=True)
class Placement:
    position: int
    candidate_id: str
    group: str


@dataclass(frozen=True)
class Allocation:
    status: str
    objective_value: float
    placements: tuple[Placement, ...]


def solve_assignment(
    candidates: list[Candidate],
    quotas: list[Quota],
    position_weights: list[int],
    max_time_seconds: float = 0.25,
) -> Allocation | None:
    """Solve a small hard-quota assignment with deterministic CP-SAT settings."""
    if cp_model is None:
        raise RuntimeError(
            "OR-Tools is required for CP-SAT; install requirements.txt first"
        )

    model = cp_model.CpModel()
    variables: dict[tuple[int, int], cp_model.IntVar] = {}

    for candidate_index, candidate in enumerate(candidates):
        for position in candidate.allowed_positions:
            if 0 <= position < len(position_weights):
                variables[candidate_index, position] = model.NewBoolVar(
                    f"x_{candidate_index}_{position}"
                )

    for position in range(len(position_weights)):
        model.AddAtMostOne(
            variable
            for (candidate_index, variable_position), variable in variables.items()
            if variable_position == position
        )

    for candidate_index in range(len(candidates)):
        model.AddAtMostOne(
            variable
            for (variable_candidate, _), variable in variables.items()
            if variable_candidate == candidate_index
        )

    for quota in quotas:
        group_variables = [
            variable
            for (candidate_index, _), variable in variables.items()
            if candidates[candidate_index].group == quota.group
        ]
        model.Add(sum(group_variables) >= quota.minimum)
        if quota.maximum is not None:
            model.Add(sum(group_variables) <= quota.maximum)

    objective_terms = []
    for (candidate_index, position), variable in variables.items():
        scaled_utility = round(candidates[candidate_index].utility * 1_000)
        objective_terms.append(scaled_utility * position_weights[position] * variable)
    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    placements = []
    for (candidate_index, position), variable in variables.items():
        if solver.Value(variable):
            candidate = candidates[candidate_index]
            placements.append(Placement(position, candidate.candidate_id, candidate.group))

    return Allocation(
        status=solver.StatusName(status),
        objective_value=solver.ObjectiveValue() / 1_000,
        placements=tuple(sorted(placements, key=lambda placement: placement.position)),
    )
