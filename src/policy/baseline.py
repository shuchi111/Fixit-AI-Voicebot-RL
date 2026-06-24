"""Behaviour policy estimated from logged trajectories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.models import Action, Transition, Trajectory
from src.policy.discretizer import StateDiscretizer


@dataclass
class BehaviorPolicy:
    """Empirical action distribution pi_b(a|s) per discretised state."""

    action_counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(dict))
    smoothing: float = 0.1

    def update(self, state_key: str, action: Action) -> None:
        bucket = self.action_counts[state_key]
        bucket[action.value] = bucket.get(action.value, 0) + 1

    def fit(self, trajectories: list[Trajectory], discretizer: StateDiscretizer) -> BehaviorPolicy:
        for trajectory in trajectories:
            for transition in trajectory.transitions:
                if transition.action is None:
                    continue
                key = discretizer.key(transition.state)
                self.update(key, transition.action)
        return self

    def action_probs(self, state_key: str) -> dict[str, float]:
        counts = self.action_counts.get(state_key, {})
        total = sum(counts.values())
        actions = [a.value for a in Action]
        if total == 0:
            uniform = 1.0 / len(actions)
            return {action: uniform for action in actions}
        smoothed_total = total + self.smoothing * len(actions)
        return {
            action: (counts.get(action, 0) + self.smoothing) / smoothed_total
            for action in actions
        }

    def propensity(self, state_key: str, action: Action) -> float:
        return self.action_probs(state_key).get(action.value, 0.0)

    def sample_action(self, state_key: str) -> Action:
        probs = self.action_probs(state_key)
        best = max(probs, key=probs.get)
        return Action(best)

    def to_dict(self) -> dict:
        return {
            "smoothing": self.smoothing,
            "action_counts": self.action_counts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorPolicy:
        policy = cls(smoothing=float(data.get("smoothing", 0.1)))
        policy.action_counts = defaultdict(
            dict, {k: dict(v) for k, v in data.get("action_counts", {}).items()}
        )
        return policy


def collect_transitions(trajectories: list[Trajectory]) -> list[Transition]:
    return [t for traj in trajectories for t in traj.transitions]
