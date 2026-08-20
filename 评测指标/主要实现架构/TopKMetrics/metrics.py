from math import log2


def user_metrics(ranked_items: list[int], relevant_items: set[int], top_k: int) -> dict[str, float]:
    top_items = ranked_items[:top_k]
    hits = [1 if item in relevant_items else 0 for item in top_items]
    hit_count = sum(hits)
    precision = hit_count / top_k if top_k > 0 else 0.0
    recall = hit_count / len(relevant_items) if relevant_items else 0.0
    hit_rate = float(hit_count > 0)
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)
    dcg = sum(hit / log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant_items), top_k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "hit_rate": hit_rate,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


def macro_average(
    recommendations: dict[int, list[int]], ground_truth: dict[int, set[int]], top_k: int
) -> dict[str, float]:
    results = [
        user_metrics(recommendations[user_id], relevant, top_k)
        for user_id, relevant in ground_truth.items()
        if relevant and user_id in recommendations
    ]
    if not results:
        return {name: 0.0 for name in ("precision", "recall", "hit_rate", "mrr", "ndcg")}
    return {
        name: sum(result[name] for result in results) / len(results)
        for name in results[0]
    }