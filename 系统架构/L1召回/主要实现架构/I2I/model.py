from collections import Counter, defaultdict
from math import sqrt


class ItemToItemCF:
    def __init__(self, window_size: int = 2) -> None:
        self.window_size = window_size
        self.neighbors: dict[int, list[tuple[int, float]]] = {}

    def fit(self, sessions: list[list[int]]) -> None:
        item_counts: Counter[int] = Counter()
        pair_counts: dict[int, Counter[int]] = defaultdict(Counter)

        for session in sessions:
            item_counts.update(set(session))
            for index, source in enumerate(session):
                left = max(0, index - self.window_size)
                right = min(len(session), index + self.window_size + 1)
                for target in session[left:right]:
                    if source != target:
                        pair_counts[source][target] += 1

        for source, targets in pair_counts.items():
            scored = [
                (target, count / sqrt(item_counts[source] * item_counts[target]))
                for target, count in targets.items()
            ]
            self.neighbors[source] = sorted(scored, key=lambda pair: pair[1], reverse=True)

    def recommend(self, seed_items: list[int], top_k: int = 10) -> list[tuple[int, float]]:
        scores: Counter[int] = Counter()
        seed_set = set(seed_items)
        for rank, seed in enumerate(reversed(seed_items), start=1):
            recency_weight = 1.0 / rank
            for candidate, similarity in self.neighbors.get(seed, []):
                if candidate not in seed_set:
                    scores[candidate] += recency_weight * similarity
        return scores.most_common(top_k)