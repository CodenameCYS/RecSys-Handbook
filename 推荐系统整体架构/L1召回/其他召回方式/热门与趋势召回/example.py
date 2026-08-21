from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class Event:
    item_id: int
    event_type: str
    timestamp: float
    scene: str
    region: str


def build_rankings(
    events: list[Event], now: float, decay_seconds: float, event_weights: dict[str, float]
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    scores: dict[tuple[str, str], dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for event in events:
        age = max(0.0, now - event.timestamp)
        score = event_weights.get(event.event_type, 0.0) * exp(-age / decay_seconds)
        for key in ((event.scene, event.region), (event.scene, "*"), ("*", "*")):
            scores[key][event.item_id] += score
    return {
        key: sorted(items.items(), key=lambda pair: (-pair[1], pair[0]))
        for key, items in scores.items()
    }


def recall(
    rankings: dict[tuple[str, str], list[tuple[int, float]]],
    scene: str,
    region: str,
    top_k: int,
) -> list[tuple[int, float, str]]:
    result: list[tuple[int, float, str]] = []
    seen: set[int] = set()
    for key in ((scene, region), (scene, "*"), ("*", "*")):
        for item_id, score in rankings.get(key, []):
            if item_id not in seen:
                result.append((item_id, round(score, 4), f"bucket={key}"))
                seen.add(item_id)
            if len(result) == top_k:
                return result
    return result


if __name__ == "__main__":
    current_time = 10_000.0
    sample_events = [
        Event(101, "click", 9_980, "feed", "east"),
        Event(101, "purchase", 9_970, "feed", "east"),
        Event(102, "click", 9_990, "feed", "west"),
        Event(103, "view", 9_995, "feed", "east"),
        Event(104, "click", 9_960, "search", "east"),
    ]
    rankings = build_rankings(
        sample_events,
        now=current_time,
        decay_seconds=300,
        event_weights={"view": 0.2, "click": 1.0, "purchase": 3.0},
    )
    print(recall(rankings, scene="feed", region="east", top_k=3))