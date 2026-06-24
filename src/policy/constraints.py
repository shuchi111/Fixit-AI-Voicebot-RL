"""Conservative constraints when improving over the behaviour policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.models import Action
from src.policy.baseline import BehaviorPolicy


@dataclass(frozen=True)
class PolicyConstraints:
    max_kl_divergence: float = 0.5
    blend_alpha: float = 0.5
    min_q_improvement: float = 0.15


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    total = 0.0
    for action, p_val in p.items():
        q_val = max(q.get(action, 1e-12), 1e-12)
        if p_val > 0:
            total += p_val * math.log(p_val / q_val)
    return total


def greedy_action_from_q(
    state_key: str,
    q_values: dict[str, dict[str, float]],
    *,
    default: Action = Action.CLARIFY_REPEAT,
) -> Action:
    state_q = q_values.get(state_key)
    if not state_q:
        return default
    best_action = max(state_q, key=state_q.get)
    return Action(best_action)


def constrained_action(
    state_key: str,
    q_values: dict[str, dict[str, float]],
    behavior_policy: BehaviorPolicy,
    constraints: PolicyConstraints,
) -> Action:
    """Pick greedy Q action only when it clearly beats the behaviour policy."""
    behavior_action = behavior_policy.sample_action(state_key)
    greedy = greedy_action_from_q(state_key, q_values)
    if greedy is behavior_action:
        return greedy

    state_q = q_values.get(state_key, {})
    q_greedy = state_q.get(greedy.value, 0.0)
    q_behavior = state_q.get(behavior_action.value, 0.0)
    if q_greedy < q_behavior + constraints.min_q_improvement:
        return behavior_action

    behavior_probs = behavior_policy.action_probs(state_key)
    greedy_probs = {a.value: 0.0 for a in Action}
    greedy_probs[greedy.value] = 1.0
    if kl_divergence(greedy_probs, behavior_probs) <= constraints.max_kl_divergence:
        return greedy
    return behavior_action
