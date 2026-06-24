"""High-level offline evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.ope import run_ope
from src.eval.propensity import PropensityConfig
from src.models import Trajectory
from src.policy.baseline import BehaviorPolicy
from src.policy.discretizer import StateDiscretizer
from src.policy.learner import LearnedPolicy
from src.split import DataSplit, split_trajectories
from src.train import load_policy


def evaluate_policies(
    trajectories: list[Trajectory],
    data_split: DataSplit,
    evaluation_policy: LearnedPolicy,
    discretizer: StateDiscretizer,
    *,
    ope_config: PropensityConfig | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    train_traj, _, test_traj = split_trajectories(trajectories, data_split)
    behavior_policy = evaluation_policy.behavior_policy or BehaviorPolicy().fit(
        train_traj, discretizer
    )
    if evaluation_policy.behavior_policy is None:
        evaluation_policy.behavior_policy = behavior_policy

    results = run_ope(
        train_traj,
        test_traj,
        behavior_policy,
        evaluation_policy,
        discretizer,
        config=ope_config,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    dr = results["dr"]
    ips = results["ips"]
    return {
        "primary_estimator": "DoublyRobust",
        "baseline_test_return": dr.baseline_behavior_return,
        "ips": ips.to_dict(),
        "dr": dr.to_dict(),
        "improvement_vs_baseline": dr.point_estimate - dr.baseline_behavior_return,
        "significant_improvement": dr.ci_95_low > dr.baseline_behavior_return,
        "n_test_episodes": dr.n_episodes,
    }


def load_evaluation_policy(
    policy_path: Path,
    *,
    fallback_discretizer: StateDiscretizer | None = None,
) -> LearnedPolicy:
    if policy_path.exists():
        return load_policy(policy_path)
    raise FileNotFoundError(
        f"Policy not found at {policy_path}. Run --train first."
    )
