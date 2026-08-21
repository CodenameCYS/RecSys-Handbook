from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecallRule:
    rule_id: str
    scene: str
    regions: frozenset[str]
    item_ids: tuple[int, ...]
    start_time: int
    end_time: int
    priority: int
    quota: int


@dataclass(frozen=True)
class Candidate:
    item_id: int
    rule_ids: tuple[str, ...]
    priority: int


def recall(
    rules: list[RecallRule],
    scene: str,
    region: str,
    now: int,
    eligible_items: set[int],
    top_k: int,
) -> list[Candidate]:
    matched = [
        rule
        for rule in rules
        if rule.scene == scene
        and region in rule.regions
        and rule.start_time <= now < rule.end_time
    ]
    selected: dict[int, tuple[list[str], int]] = {}
    for rule in sorted(matched, key=lambda value: (-value.priority, value.rule_id)):
        accepted = 0
        for item_id in rule.item_ids:
            if item_id not in eligible_items:
                continue
            rule_ids, priority = selected.setdefault(item_id, ([], rule.priority))
            rule_ids.append(rule.rule_id)
            selected[item_id] = (rule_ids, max(priority, rule.priority))
            accepted += 1
            if accepted == rule.quota:
                break
    candidates = [
        Candidate(item_id, tuple(rule_ids), priority)
        for item_id, (rule_ids, priority) in selected.items()
    ]
    return sorted(candidates, key=lambda value: (-value.priority, value.item_id))[:top_k]


if __name__ == "__main__":
    sample_rules = [
        RecallRule("festival-v2", "feed", frozenset({"east"}), (301, 302), 0, 200, 20, 2),
        RecallRule("sports-live-v1", "feed", frozenset({"east", "west"}), (302, 303), 50, 120, 30, 1),
    ]
    print(recall(sample_rules, "feed", "east", now=100, eligible_items={301, 302}, top_k=3))