from __future__ import annotations

import math
from collections import Counter


Interactions = dict[int, dict[int, float]]
NeighborTable = dict[int, list[tuple[int, float]]]


def build_user_neighbors(
    interactions: Interactions,
    top_k: int = 50,
    shrinkage: float = 10.0,
) -> NeighborTable:
    item_users: dict[int, list[int]] = {}
    for user_id, item_weights in interactions.items():
        for item_id in item_weights:
            item_users.setdefault(item_id, []).append(user_id)

    cooccurrence: dict[int, Counter[int]] = {
        user_id: Counter() for user_id in interactions
    }
    for users in item_users.values():
        inverse_popularity = 1.0 / math.log1p(len(users))
        for user_id in users:
            for neighbor_id in users:
                if user_id != neighbor_id:
                    cooccurrence[user_id][neighbor_id] += inverse_popularity

    activity = {user_id: len(items) for user_id, items in interactions.items()}
    neighbors: NeighborTable = {}
    for user_id, counts in cooccurrence.items():
        similarities = []
        for neighbor_id, weighted_overlap in counts.items():
            cosine = weighted_overlap / math.sqrt(
                activity[user_id] * activity[neighbor_id]
            )
            common_items = len(
                interactions[user_id].keys() & interactions[neighbor_id].keys()
            )
            reliability = common_items / (common_items + shrinkage)
            similarities.append((neighbor_id, cosine * reliability))
        neighbors[user_id] = sorted(
            similarities, key=lambda pair: pair[1], reverse=True
        )[:top_k]
    return neighbors


def recommend(
    interactions: Interactions,
    neighbors: NeighborTable,
    user_id: int,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    history = interactions.get(user_id, {})
    scores: Counter[int] = Counter()
    for neighbor_id, similarity in neighbors.get(user_id, []):
        for item_id, behavior_weight in interactions[neighbor_id].items():
            if item_id not in history:
                scores[item_id] += similarity * behavior_weight
    return scores.most_common(top_k)