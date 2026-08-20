from math import sqrt


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    return dot_product / max(left_norm * right_norm, 1e-12)


def mmr_rerank(
    candidates: list[dict[str, object]], top_k: int, relevance_weight: float = 0.7
) -> list[dict[str, object]]:
    remaining = list(candidates)
    selected: list[dict[str, object]] = []

    while remaining and len(selected) < top_k:
        def mmr_score(candidate: dict[str, object]) -> float:
            relevance = float(candidate["score"])
            vector = list(candidate["vector"])
            redundancy = max(
                (cosine_similarity(vector, list(item["vector"])) for item in selected),
                default=0.0,
            )
            return relevance_weight * relevance - (1.0 - relevance_weight) * redundancy

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected