"""Fitted Q Evaluation — learn Q_hat from behaviour-policy trajectories."""

from __future__ import annotations

from collections import defaultdict

from src.models import Action, Trajectory
from src.policy.discretizer import StateDiscretizer


class FittedQEstimator:
    """Tabular Q function fitted on logged behaviour data."""

    def __init__(self, gamma: float = 0.95, alpha: float = 0.1, epochs: int = 30):
        self.gamma = gamma
        self.alpha = alpha
        self.epochs = epochs
        self.q_values: dict[str, dict[str, float]] = defaultdict(dict)

    def predict(self, state_key: str, action: Action) -> float:
        return self.q_values.get(state_key, {}).get(action.value, 0.0)

    def max_q(self, state_key: str) -> float:
        state_q = self.q_values.get(state_key)
        if not state_q:
            return 0.0
        return max(state_q.values())

    def fit(
        self,
        trajectories: list[Trajectory],
        discretizer: StateDiscretizer,
    ) -> FittedQEstimator:
        for _ in range(self.epochs):
            for trajectory in trajectories:
                for transition in trajectory.transitions:
                    if transition.action is None:
                        continue
                    state_key = discretizer.key(transition.state)
                    action = transition.action
                    current = self.predict(state_key, action)
                    if transition.done or transition.next_state is None:
                        target = transition.reward
                    else:
                        next_key = discretizer.key(transition.next_state)
                        target = transition.reward + self.gamma * self.max_q(
                            next_key
                        )
                    updated = current + self.alpha * (target - current)
                    self.q_values[state_key][action.value] = updated
        return self
