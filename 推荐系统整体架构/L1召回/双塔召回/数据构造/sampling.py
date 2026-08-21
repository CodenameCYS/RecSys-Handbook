from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class NegativeSample:
    item_id: int
    probability: float
    source: str


class NegativeSampler(Protocol):
    def sample(
        self, positive_item: int, excluded_items: set[int], count: int
    ) -> list[NegativeSample]: ...


class UniformSampler:
    def __init__(self, item_ids: list[int], seed: int = 7) -> None:
        self.item_ids = item_ids
        self.random = random.Random(seed)

    def sample(
        self, positive_item: int, excluded_items: set[int], count: int
    ) -> list[NegativeSample]:
        excluded = excluded_items | {positive_item}
        candidates = [item for item in self.item_ids if item not in excluded]
        if count > len(candidates):
            raise ValueError("count exceeds the available negative pool")
        probability = 1.0 / len(candidates)
        return [
            NegativeSample(item, probability, "uniform")
            for item in self.random.sample(candidates, count)
        ]


class PopularitySampler:
    def __init__(
        self, item_frequencies: dict[int, int], alpha: float = 0.75, seed: int = 7
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.item_frequencies = item_frequencies
        self.alpha = alpha
        self.random = random.Random(seed)

    def sample(
        self, positive_item: int, excluded_items: set[int], count: int
    ) -> list[NegativeSample]:
        excluded = excluded_items | {positive_item}
        candidates = [
            item for item in self.item_frequencies if item not in excluded
        ]
        weights = [self.item_frequencies[item] ** self.alpha for item in candidates]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        chosen = self.random.choices(candidates, weights=weights, k=count)
        probability_by_item = dict(zip(candidates, probabilities, strict=True))
        return [
            NegativeSample(item, probability_by_item[item], "popularity")
            for item in chosen
        ]


class ExposureSampler:
    def __init__(self, seed: int = 7) -> None:
        self.random = random.Random(seed)

    def sample_from_exposures(
        self, exposed_items: list[int], positive_items: set[int], count: int
    ) -> list[NegativeSample]:
        candidates = sorted(set(exposed_items) - positive_items)
        if not candidates:
            return []
        selected = self.random.sample(candidates, min(count, len(candidates)))
        probability = 1.0 / len(candidates)
        return [
            NegativeSample(item, probability, "exposure") for item in selected
        ]


def corrected_logits(
    scores: dict[int, float], samples: list[NegativeSample]
) -> dict[int, float]:
    probability_by_item = {
        item: sum(sample.probability for sample in item_samples)
        / len(item_samples)
        for item, item_samples in _group_samples(samples).items()
    }
    return {
        item: score - math.log(probability_by_item.get(item, 1.0))
        for item, score in scores.items()
    }


def sample_histogram(samples: list[NegativeSample]) -> Counter[int]:
    return Counter(sample.item_id for sample in samples)


def _group_samples(samples: list[NegativeSample]) -> dict[int, list[NegativeSample]]:
    grouped: dict[int, list[NegativeSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.item_id, []).append(sample)
    return grouped