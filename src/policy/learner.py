"""Tabular Q-learning over discretised dialogue states."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.control.freeze import FreezeRules, apply_freeze_rules
from src.models import Action, Trajectory
from src.policy.baseline import BehaviorPolicy
from src.policy.constraints import PolicyConstraints, constrained_action
from src.policy.discretizer import StateDiscretizer


@dataclass
class QLearningConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epochs: int = 50

    @classmethod
    def from_dict(cls, data: dict) -> QLearningConfig:
        return cls(
            alpha=float(data.get("alpha", 0.1)),
            gamma=float(data.get("gamma", 0.95)),
            epochs=int(data.get("epochs", 50)),
        )


@dataclass
class LearnedPolicy:
    """Tabular Q-policy with behaviour-policy constraints."""

    q_values: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    discretizer: StateDiscretizer = field(default_factory=StateDiscretizer)
    constraints: PolicyConstraints = field(default_factory=PolicyConstraints)
    behavior_policy: BehaviorPolicy | None = None
    freeze_rules: FreezeRules | None = None
    version: str = "v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def q(self, state_key: str, action: Action) -> float:
        return self.q_values.get(state_key, {}).get(action.value, 0.0)

    def max_q(self, state_key: str) -> float:
        state_q = self.q_values.get(state_key)
        if not state_q:
            return 0.0
        return max(state_q.values())

    def select_action(self, state_key: str) -> Action:
        if self.behavior_policy is None:
            proposed = constrained_action(
                state_key,
                self.q_values,
                BehaviorPolicy(),
                self.constraints,
            )
        else:
            proposed = constrained_action(
                state_key,
                self.q_values,
                self.behavior_policy,
                self.constraints,
            )
        action, _ = apply_freeze_rules(state_key, proposed, self.freeze_rules)
        return action

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "q_values": {k: dict(v) for k, v in self.q_values.items()},
            "discretizer": self.discretizer.to_dict(),
            "constraints": {
                "max_kl_divergence": self.constraints.max_kl_divergence,
                "blend_alpha": self.constraints.blend_alpha,
                "min_q_improvement": self.constraints.min_q_improvement,
            },
            "behavior_policy": (
                self.behavior_policy.to_dict() if self.behavior_policy else None
            ),
            "freeze_rules": (
                self.freeze_rules.to_dict() if self.freeze_rules else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPolicy:
        q_values: dict[str, dict[str, float]] = defaultdict(dict)
        for state_key, actions in data.get("q_values", {}).items():
            q_values[state_key] = {k: float(v) for k, v in actions.items()}
        constraints_data = data.get("constraints", {})
        behavior_data = data.get("behavior_policy")
        freeze_data = data.get("freeze_rules")
        return cls(
            q_values=q_values,
            discretizer=StateDiscretizer.from_dict(data.get("discretizer", {})),
            constraints=PolicyConstraints(
                max_kl_divergence=float(
                    constraints_data.get("max_kl_divergence", 0.5)
                ),
                blend_alpha=float(constraints_data.get("blend_alpha", 0.5)),
                min_q_improvement=float(
                    constraints_data.get("min_q_improvement", 0.15)
                ),
            ),
            behavior_policy=(
                BehaviorPolicy.from_dict(behavior_data) if behavior_data else None
            ),
            freeze_rules=(
                FreezeRules.from_dict(freeze_data) if freeze_data else None
            ),
            version=str(data.get("version", "v1")),
            metadata=dict(data.get("metadata", {})),
        )


def _set_q(
    q_values: dict[str, dict[str, float]],
    state_key: str,
    action: Action,
    value: float,
) -> None:
    q_values[state_key][action.value] = value


def train_q_learning(
    train_trajectories: list[Trajectory],
    *,
    discretizer: StateDiscretizer,
    behavior_policy: BehaviorPolicy,
    config: QLearningConfig,
    constraints: PolicyConstraints | None = None,
) -> LearnedPolicy:
    q_values: dict[str, dict[str, float]] = defaultdict(dict)
    constraints = constraints or PolicyConstraints()

    for _ in range(config.epochs):
        for trajectory in train_trajectories:
            for transition in trajectory.transitions:
                if transition.action is None:
                    continue
                state_key = discretizer.key(transition.state)
                action = transition.action
                current_q = q_values.get(state_key, {}).get(action.value, 0.0)
                if transition.done or transition.next_state is None:
                    target = transition.reward
                else:
                    next_key = discretizer.key(transition.next_state)
                    target = transition.reward + config.gamma * max(
                        q_values.get(next_key, {}).values() or [0.0]
                    )
                updated = current_q + config.alpha * (target - current_q)
                _set_q(q_values, state_key, action, updated)

    policy = LearnedPolicy(
        q_values=q_values,
        discretizer=discretizer,
        constraints=constraints,
        behavior_policy=behavior_policy,
        metadata={
            "epochs": config.epochs,
            "alpha": config.alpha,
            "gamma": config.gamma,
        },
    )
    return policy
