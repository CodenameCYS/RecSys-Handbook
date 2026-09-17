from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    item_id: str
    quality: float
    vector: list[float]
    input_index: int


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def determinant(matrix: list[list[float]]) -> float:
    """Compute a small matrix determinant with Gaussian elimination."""
    working = [row[:] for row in matrix]
    value = 1.0
    for column in range(len(working)):
        pivot = max(range(column, len(working)), key=lambda row: abs(working[row][column]))
        if abs(working[pivot][column]) < 1e-12:
            return 0.0
        if pivot != column:
            working[column], working[pivot] = working[pivot], working[column]
            value *= -1.0
        pivot_value = working[column][column]
        value *= pivot_value
        for row in range(column + 1, len(working)):
            ratio = working[row][column] / pivot_value
            for remaining_column in range(column + 1, len(working)):
                working[row][remaining_column] -= ratio * working[column][remaining_column]
    return value


def subset_determinant(candidates: list[Candidate]) -> float:
    kernel = [
        [
            left.quality * right.quality * dot_product(left.vector, right.vector)
            for right in candidates
        ]
        for left in candidates
    ]
    return max(determinant(kernel), 0.0)


def dpp_greedy_rerank(candidates: list[Candidate], top_k: int) -> list[Candidate]:
    """Select a small quality-diverse set by greedily maximizing its determinant."""
    remaining = list(candidates)
    selected: list[Candidate] = []
    while remaining and len(selected) < top_k:
        best = max(
            remaining,
            key=lambda candidate: (
                subset_determinant([*selected, candidate]),
                -candidate.input_index,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected