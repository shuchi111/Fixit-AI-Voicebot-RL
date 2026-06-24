"""Evaluate policies by replaying transitions from logged trajectories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from src.models import Action, Transition, Trajectory
from src.policy.baseline import BehaviorPolicy
from src.policy.discretizer import StateDiscretizer
from src.policy.learner import LearnedPolicy


@dataclass
class ReplayEvalResult:
    mean_return: float
    num_episodes: int
    matched_action_rate: float

    def to_dict(self) -> dict:
        return {
            "mean_return": self.mean_return,
            "num_episodes": self.num_episodes,
            "matched_action_rate": self.matched_action_rate,
        }


def build_reward_lookup(
    trajectories: list[Trajectory],
    discretizer: StateDiscretizer,
) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            if transition.action is None:
                continue
            key = (
                discretizer.key(transition.state),
                transition.action.value,
            )
            totals[key] += transition.reward
            counts[key] += 1
    return {key: totals[key] / counts[key] for key in counts}


def evaluate_behavior_returns(trajectories: list[Trajectory]) -> ReplayEvalResult:
    returns = [traj.total_return for traj in trajectories]
    if not returns:
        return ReplayEvalResult(0.0, 0, 0.0)
    return ReplayEvalResult(
        mean_return=sum(returns) / len(returns),
        num_episodes=len(returns),
        matched_action_rate=1.0,
    )


def evaluate_learned_policy(
    trajectories: list[Trajectory],
    policy: LearnedPolicy,
    reward_lookup: dict[tuple[str, str], float],
) -> ReplayEvalResult:
    """
    Pessimistic offline replay: keep the observed reward unless the learned
    action has a higher average reward in training data for the same state cluster.
    """
    discretizer = policy.discretizer
    episode_returns: list[float] = []
    matched = 0
    improved = 0
    total_steps = 0

    for trajectory in trajectories:
        ep_return = 0.0
        for transition in trajectory.transitions:
            if transition.action is None:
                continue
            state_key = discretizer.key(transition.state)
            learned_action = policy.select_action(state_key)
            total_steps += 1
            if learned_action is transition.action:
                matched += 1
                ep_return += transition.reward
                continue

            lookup_key = (state_key, learned_action.value)
            candidate = reward_lookup.get(lookup_key)
            if candidate is not None and candidate > transition.reward:
                improved += 1
                ep_return += candidate
            else:
                ep_return += transition.reward
        episode_returns.append(ep_return)

    if not episode_returns:
        return ReplayEvalResult(0.0, 0, 0.0)
    return ReplayEvalResult(
        mean_return=sum(episode_returns) / len(episode_returns),
        num_episodes=len(episode_returns),
        matched_action_rate=(matched + improved) / max(total_steps, 1),
    )


def evaluate_behavior_policy_model(
    trajectories: list[Trajectory],
    behavior_policy: BehaviorPolicy,
    reward_lookup: dict[tuple[str, str], float],
    discretizer: StateDiscretizer,
) -> ReplayEvalResult:
    """Estimate return if behaviour policy were followed via reward lookup."""
    episode_returns: list[float] = []
    matched = 0
    total_steps = 0

    for trajectory in trajectories:
        ep_return = 0.0
        for transition in trajectory.transitions:
            if transition.action is None:
                continue
            state_key = discretizer.key(transition.state)
            behavior_action = behavior_policy.sample_action(state_key)
            total_steps += 1
            if behavior_action is transition.action:
                matched += 1
                ep_return += transition.reward
            else:
                lookup_key = (state_key, behavior_action.value)
                ep_return += reward_lookup.get(lookup_key, 0.0)
        episode_returns.append(ep_return)

    return ReplayEvalResult(
        mean_return=sum(episode_returns) / len(episode_returns),
        num_episodes=len(episode_returns),
        matched_action_rate=matched / max(total_steps, 1),
    )
