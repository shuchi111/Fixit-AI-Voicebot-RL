"""Propensity scoring for off-policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Action
from src.policy.baseline import BehaviorPolicy
from src.policy.learner import LearnedPolicy


@dataclass(frozen=True)
class PropensityConfig:
    min_propensity: float = 0.01
    rho_max: float = 10.0


def behavior_propensity(
    behavior_policy: BehaviorPolicy,
    state_key: str,
    action: Action,
    *,
    config: PropensityConfig,
) -> float:
    return max(behavior_policy.propensity(state_key, action), config.min_propensity)


def evaluation_propensity(
    evaluation_policy: LearnedPolicy,
    state_key: str,
    action: Action,
) -> float:
    """Deterministic evaluation policy: mass 1 on the selected action."""
    selected = evaluation_policy.select_action(state_key)
    return 1.0 if selected is action else 0.0


def importance_ratio(
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    state_key: str,
    action: Action,
    *,
    config: PropensityConfig,
) -> float:
    pi_b = behavior_propensity(
        behavior_policy, state_key, action, config=config
    )
    pi_e = evaluation_propensity(evaluation_policy, state_key, action)
    return min(pi_e / pi_b, config.rho_max)
