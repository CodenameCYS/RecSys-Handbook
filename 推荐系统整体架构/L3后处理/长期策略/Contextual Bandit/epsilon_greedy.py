from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence


@dataclass(frozen=True)
class ActionEstimate:
    action_id: str
    estimated_reward: float


@dataclass(frozen=True)
class Decision:
    action_id: str
    propensity: float
    policy_mode: str


def choose_action(
    eligible_actions: Sequence[ActionEstimate], epsilon: float, random_source: Random
) -> Decision:
    """Select only from caller-supplied eligible actions and log its exact propensity."""
    if not eligible_actions:
        raise ValueError("eligible_actions must not be empty")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")

    best_reward = max(action.estimated_reward for action in eligible_actions)
    greedy_actions = [action for action in eligible_actions if action.estimated_reward == best_reward]
    greedy_actions.sort(key=lambda action: action.action_id)
    greedy_action = greedy_actions[0]
    action_count = len(eligible_actions)

    if random_source.random() < epsilon:
        selected_action = eligible_actions[random_source.randrange(action_count)]
        policy_mode = "explore"
    else:
        selected_action = greedy_action
        policy_mode = "exploit"

    propensity = epsilon / action_count
    if selected_action.action_id == greedy_action.action_id:
        propensity += 1.0 - epsilon
    return Decision(
        action_id=selected_action.action_id,
        propensity=propensity,
        policy_mode=policy_mode,
    )