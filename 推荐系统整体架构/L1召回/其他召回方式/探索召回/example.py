from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    item_id: int
    estimated_reward: float
    eligible: bool = True
    remaining_budget: int = 1


@dataclass(frozen=True)
class Selection:
    item_id: int
    propensity: float
    decision: str


def epsilon_greedy_recall(
    arms: list[Arm], top_k: int, epsilon: float, random_source: random.Random
) -> list[Selection]:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    available = [arm for arm in arms if arm.eligible and arm.remaining_budget > 0]
    result: list[Selection] = []
    while available and len(result) < top_k:
        best = max(available, key=lambda arm: (arm.estimated_reward, -arm.item_id))
        exploring = random_source.random() < epsilon
        chosen = random_source.choice(available) if exploring else best
        random_probability = epsilon / len(available)
        propensity = random_probability + (1.0 - epsilon if chosen == best else 0.0)
        result.append(
            Selection(chosen.item_id, round(propensity, 6), "explore" if exploring else "exploit")
        )
        available.remove(chosen)
    return result


if __name__ == "__main__":
    candidate_arms = [
        Arm(401, estimated_reward=0.18, remaining_budget=10),
        Arm(402, estimated_reward=0.12, remaining_budget=3),
        Arm(403, estimated_reward=0.08, remaining_budget=5),
        Arm(404, estimated_reward=0.30, eligible=False, remaining_budget=10),
    ]
    print(epsilon_greedy_recall(candidate_arms, top_k=2, epsilon=0.2, random_source=random.Random(7)))