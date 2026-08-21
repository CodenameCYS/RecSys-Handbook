from __future__ import annotations

from collections import defaultdict


Graph = dict[str, list[tuple[str, float]]]


def personalized_pagerank(
    graph: Graph,
    seeds: dict[str, float],
    restart_probability: float = 0.25,
    iterations: int = 30,
) -> dict[str, float]:
    seed_total = sum(seeds.values())
    restart = {node: weight / seed_total for node, weight in seeds.items()}
    scores = restart.copy()
    for _ in range(iterations):
        updated = defaultdict(float)
        for node, probability in restart.items():
            updated[node] += restart_probability * probability
        for source, probability in scores.items():
            neighbors = graph.get(source, [])
            weight_sum = sum(weight for _, weight in neighbors)
            if not neighbors or weight_sum == 0:
                for node, seed_probability in restart.items():
                    updated[node] += (1 - restart_probability) * probability * seed_probability
                continue
            for target, weight in neighbors:
                updated[target] += (1 - restart_probability) * probability * weight / weight_sum
        scores = dict(updated)
    return scores


def recall_items(
    graph: Graph, seeds: dict[str, float], consumed: set[str], top_k: int
) -> list[tuple[str, float]]:
    scores = personalized_pagerank(graph, seeds)
    candidates = [
        (node, round(score, 6))
        for node, score in scores.items()
        if node.startswith("item:") and node not in consumed
    ]
    return sorted(candidates, key=lambda pair: (-pair[1], pair[0]))[:top_k]


if __name__ == "__main__":
    sample_graph: Graph = {
        "user:1": [("item:1", 1.0), ("topic:ai", 0.8)],
        "item:1": [("user:1", 1.0), ("author:a", 1.0), ("topic:ai", 1.0)],
        "author:a": [("item:1", 1.0), ("item:2", 1.0)],
        "topic:ai": [("item:1", 1.0), ("item:2", 0.8), ("item:3", 0.6)],
        "item:2": [("author:a", 1.0), ("topic:ai", 0.8)],
        "item:3": [("topic:ai", 1.0)],
    }
    print(recall_items(sample_graph, {"user:1": 1.0}, {"item:1"}, top_k=2))