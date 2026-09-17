from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    item_id: str
    score: float
    intent_scores: dict[str, float]
    input_index: int


def xquad_rerank(
    candidates: list[Candidate],
    intent_weights: dict[str, float],
    top_k: int,
    diversity_weight: float = 0.4,
) -> list[Candidate]:
    """Greedily balance candidate utility and unfulfilled intent coverage."""
    remaining = list(candidates)
    selected: list[Candidate] = []

    while remaining and len(selected) < top_k:
        def xquad_score(candidate: Candidate) -> float:
            coverage_gain = 0.0
            for intent, weight in intent_weights.items():
                uncovered_probability = 1.0
                for selected_candidate in selected:
                    uncovered_probability *= 1.0 - selected_candidate.intent_scores.get(
                        intent, 0.0
                    )
                coverage_gain += (
                    weight
                    * candidate.intent_scores.get(intent, 0.0)
                    * uncovered_probability
                )
            return (
                (1.0 - diversity_weight) * candidate.score
                + diversity_weight * coverage_gain
            )

        best = max(
            remaining,
            key=lambda candidate: (xquad_score(candidate), -candidate.input_index),
        )
        selected.append(best)
        remaining.remove(best)
    return selected