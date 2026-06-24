"""Training loop for the voicebot dialogue policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.eval.replay import (
    build_reward_lookup,
    evaluate_behavior_returns,
    evaluate_learned_policy,
)
from src.models import Trajectory
from src.policy.baseline import BehaviorPolicy
from src.policy.constraints import PolicyConstraints
from src.policy.discretizer import StateDiscretizer
from src.policy.learner import LearnedPolicy, QLearningConfig, train_q_learning
from src.split import DataSplit, split_trajectories


@dataclass
class TrainResult:
    policy: LearnedPolicy
    baseline_val_return: float
    learned_val_return: float
    learning_curve: list[dict[str, float]]
    data_split: DataSplit

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_val_return": self.baseline_val_return,
            "learned_val_return": self.learned_val_return,
            "improvement": self.learned_val_return - self.baseline_val_return,
            "learning_curve": self.learning_curve,
            "data_split": {
                "train": len(self.data_split.train),
                "val": len(self.data_split.val),
                "test": len(self.data_split.test),
            },
            "policy_metadata": self.policy.metadata,
        }


def _train_with_curve(
    train_trajectories: list[Trajectory],
    val_trajectories: list[Trajectory],
    *,
    discretizer: StateDiscretizer,
    behavior_policy: BehaviorPolicy,
    q_config: QLearningConfig,
    constraints: PolicyConstraints,
) -> tuple[LearnedPolicy, list[dict[str, float]]]:
    reward_lookup = build_reward_lookup(train_trajectories, discretizer)
    q_values: dict[str, dict[str, float]] = {}
    curve: list[dict[str, float]] = []

    for epoch in range(1, q_config.epochs + 1):
        from collections import defaultdict

        epoch_q: dict[str, dict[str, float]] = defaultdict(dict)
        if q_values:
            for state_key, actions in q_values.items():
                epoch_q[state_key] = dict(actions)

        for trajectory in train_trajectories:
            for transition in trajectory.transitions:
                if transition.action is None:
                    continue
                state_key = discretizer.key(transition.state)
                action = transition.action
                current_q = epoch_q.get(state_key, {}).get(action.value, 0.0)
                if transition.done or transition.next_state is None:
                    target = transition.reward
                else:
                    next_key = discretizer.key(transition.next_state)
                    target = transition.reward + q_config.gamma * max(
                        epoch_q.get(next_key, {}).values() or [0.0]
                    )
                updated = current_q + q_config.alpha * (target - current_q)
                epoch_q[state_key][action.value] = updated

        q_values = {k: dict(v) for k, v in epoch_q.items()}
        policy = LearnedPolicy(
            q_values=q_values,
            discretizer=discretizer,
            constraints=constraints,
            behavior_policy=behavior_policy,
            metadata={"epoch": epoch},
        )
        val_eval = evaluate_learned_policy(
            val_trajectories, policy, reward_lookup
        )
        curve.append(
            {
                "epoch": float(epoch),
                "val_return": val_eval.mean_return,
            }
        )

    final_policy = LearnedPolicy(
        q_values=q_values,
        discretizer=discretizer,
        constraints=constraints,
        behavior_policy=behavior_policy,
        metadata={
            "epochs": q_config.epochs,
            "alpha": q_config.alpha,
            "gamma": q_config.gamma,
        },
    )
    return final_policy, curve


def train_policy(
    trajectories: list[Trajectory],
    data_split: DataSplit,
    *,
    discretizer: StateDiscretizer,
    q_config: QLearningConfig,
    constraints: PolicyConstraints,
) -> TrainResult:
    train_traj, val_traj, _ = split_trajectories(trajectories, data_split)
    behavior_policy = BehaviorPolicy().fit(train_traj, discretizer)

    baseline_val = evaluate_behavior_returns(val_traj).mean_return
    policy, curve = _train_with_curve(
        train_traj,
        val_traj,
        discretizer=discretizer,
        behavior_policy=behavior_policy,
        q_config=q_config,
        constraints=constraints,
    )
    reward_lookup = build_reward_lookup(train_traj, discretizer)
    learned_val = evaluate_learned_policy(val_traj, policy, reward_lookup).mean_return

    policy.metadata.update(
        {
            "baseline_val_return": baseline_val,
            "learned_val_return": learned_val,
        }
    )
    return TrainResult(
        policy=policy,
        baseline_val_return=baseline_val,
        learned_val_return=learned_val,
        learning_curve=curve,
        data_split=data_split,
    )


def save_policy(policy: LearnedPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")


def load_policy(path: Path) -> LearnedPolicy:
    return LearnedPolicy.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_learning_curve(curve: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [point["epoch"] for point in curve]
    returns = [point["val_return"] for point in curve]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, returns, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Estimated val return")
    plt.title("Policy learning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
