"""Offline policy evaluation estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.eval.fqe import FittedQEstimator
from src.eval.propensity import PropensityConfig, importance_ratio
from src.eval.replay import evaluate_behavior_returns
from src.models import Trajectory
from src.policy.baseline import BehaviorPolicy
from src.policy.discretizer import StateDiscretizer
from src.policy.learner import LearnedPolicy


@dataclass
class OPEResult:
    estimator: str
    point_estimate: float
    ci_95_low: float
    ci_95_high: float
    baseline_behavior_return: float
    n_episodes: int
    assumptions: list[str] = field(default_factory=list)
    failure_cases: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "point_estimate": self.point_estimate,
            "ci_95": [self.ci_95_low, self.ci_95_high],
            "baseline_behavior_return": self.baseline_behavior_return,
            "n_episodes": self.n_episodes,
            "assumptions": self.assumptions,
            "failure_cases": self.failure_cases,
            "diagnostics": self.diagnostics,
        }


ASSUMPTIONS = [
    "Positivity: pi_b(a|s) > 0 whenever pi_e(a|s) > 0",
    "Sequential exchangeability: no unmeasured confounders in logged data",
    "Stationary logging policy: behaviour policy stable across the dataset",
    "Accurate Q_hat: FQE converged on training trajectories",
]

FAILURE_CASES = [
    "Evaluation policy takes actions rarely seen in pi_b support; importance "
    "weights clip at rho_max and DR extrapolates optimistically.",
    "State discretisation merges distinct contexts; Q_hat is biased within clusters.",
    "Reward is a constructed proxy; OPE optimises proxy not true call quality.",
]


def _episode_ips(
    trajectory: Trajectory,
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    discretizer: StateDiscretizer,
    config: PropensityConfig,
) -> float:
    weight = 1.0
    episode_return = trajectory.total_return
    for transition in trajectory.transitions:
        if transition.action is None:
            continue
        state_key = discretizer.key(transition.state)
        rho = importance_ratio(
            behavior_policy,
            evaluation_policy,
            state_key,
            transition.action,
            config=config,
        )
        weight *= rho
    return weight * episode_return


def _episode_dr(
    trajectory: Trajectory,
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    q_hat: FittedQEstimator,
    discretizer: StateDiscretizer,
    config: PropensityConfig,
) -> float:
    total = 0.0
    for transition in trajectory.transitions:
        if transition.action is None:
            continue
        state_key = discretizer.key(transition.state)
        action = transition.action
        eval_action = evaluation_policy.select_action(state_key)
        rho = importance_ratio(
            behavior_policy,
            evaluation_policy,
            state_key,
            action,
            config=config,
        )
        q_obs = q_hat.predict(state_key, action)
        q_eval = q_hat.predict(state_key, eval_action)
        total += rho * (transition.reward - q_obs) + q_eval
    return total


def _low_propensity_fraction(
    trajectories: list[Trajectory],
    behavior_policy: BehaviorPolicy,
    discretizer: StateDiscretizer,
    config: PropensityConfig,
) -> float:
    low = 0
    total = 0
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            if transition.action is None:
                continue
            state_key = discretizer.key(transition.state)
            pi_b = behavior_policy.propensity(state_key, transition.action)
            total += 1
            if pi_b < config.min_propensity * 2:
                low += 1
    return low / max(total, 1)


def _episode_weight(
    trajectory: Trajectory,
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    discretizer: StateDiscretizer,
    config: PropensityConfig,
) -> float:
    weight = 1.0
    for transition in trajectory.transitions:
        if transition.action is None:
            continue
        state_key = discretizer.key(transition.state)
        weight *= importance_ratio(
            behavior_policy,
            evaluation_policy,
            state_key,
            transition.action,
            config=config,
        )
    return weight


def estimate_ips(
    trajectories: list[Trajectory],
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    discretizer: StateDiscretizer,
    *,
    config: PropensityConfig | None = None,
) -> tuple[float, list[tuple[float, float]]]:
    """Return SNIPS estimate and per-episode (weight, return) pairs."""
    config = config or PropensityConfig()
    pairs: list[tuple[float, float]] = []
    for trajectory in trajectories:
        weight = _episode_weight(
            trajectory, behavior_policy, evaluation_policy, discretizer, config
        )
        pairs.append((weight, trajectory.total_return))

    weight_sum = sum(weight for weight, _ in pairs)
    if weight_sum <= 0:
        return 0.0, pairs
    estimate = sum(weight * ret for weight, ret in pairs) / weight_sum
    return estimate, pairs


def bootstrap_ips_ci(
    episode_pairs: list[tuple[float, float]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not episode_pairs:
        return 0.0, 0.0
    rng = __import__("random").Random(seed)
    n = len(episode_pairs)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        draw = [episode_pairs[rng.randrange(n)] for _ in range(n)]
        weight_sum = sum(weight for weight, _ in draw)
        if weight_sum <= 0:
            samples.append(0.0)
        else:
            samples.append(
                sum(weight * ret for weight, ret in draw) / weight_sum
            )
    samples.sort()
    low_idx = int((alpha / 2) * n_bootstrap)
    high_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return samples[low_idx], samples[min(high_idx, n_bootstrap - 1)]


def estimate_dr(
    trajectories: list[Trajectory],
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    q_hat: FittedQEstimator,
    discretizer: StateDiscretizer,
    *,
    config: PropensityConfig | None = None,
) -> tuple[float, list[float]]:
    config = config or PropensityConfig()
    per_episode = [
        _episode_dr(
            trajectory,
            behavior_policy,
            evaluation_policy,
            q_hat,
            discretizer,
            config,
        )
        for trajectory in trajectories
    ]
    if not per_episode:
        return 0.0, per_episode
    return sum(per_episode) / len(per_episode), per_episode


def bootstrap_ci(
    per_episode_estimates: list[float],
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not per_episode_estimates:
        return 0.0, 0.0
    rng = __import__("random").Random(seed)
    n = len(per_episode_estimates)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        draw = [per_episode_estimates[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low_idx = int((alpha / 2) * n_bootstrap)
    high_idx = int((1 - alpha / 2) * n_bootstrap) - 1
    return samples[low_idx], samples[min(high_idx, n_bootstrap - 1)]


def run_ope(
    train_trajectories: list[Trajectory],
    test_trajectories: list[Trajectory],
    behavior_policy: BehaviorPolicy,
    evaluation_policy: LearnedPolicy,
    discretizer: StateDiscretizer,
    *,
    config: PropensityConfig | None = None,
    fqe_gamma: float = 0.95,
    fqe_alpha: float = 0.1,
    fqe_epochs: int = 30,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, OPEResult]:
    config = config or PropensityConfig()
    q_hat = FittedQEstimator(
        gamma=fqe_gamma, alpha=fqe_alpha, epochs=fqe_epochs
    ).fit(train_trajectories, discretizer)

    baseline_return = evaluate_behavior_returns(test_trajectories).mean_return
    low_prop = _low_propensity_fraction(
        test_trajectories, behavior_policy, discretizer, config
    )

    ips_point, ips_pairs = estimate_ips(
        test_trajectories,
        behavior_policy,
        evaluation_policy,
        discretizer,
        config=config,
    )
    ips_low, ips_high = bootstrap_ips_ci(
        ips_pairs, n_bootstrap=bootstrap_samples, seed=seed
    )

    dr_point, dr_eps = estimate_dr(
        test_trajectories,
        behavior_policy,
        evaluation_policy,
        q_hat,
        discretizer,
        config=config,
    )
    dr_low, dr_high = bootstrap_ci(
        dr_eps, n_bootstrap=bootstrap_samples, seed=seed
    )

    matched = 0
    total = 0
    for trajectory in test_trajectories:
        for transition in trajectory.transitions:
            if transition.action is None:
                continue
            state_key = discretizer.key(transition.state)
            if evaluation_policy.select_action(state_key) is transition.action:
                matched += 1
            total += 1

    diagnostics = {
        "low_propensity_fraction": low_prop,
        "matched_action_rate": matched / max(total, 1),
        "ips_nonzero_weight_episodes": sum(1 for w, _ in ips_pairs if w > 0),
    }

    return {
        "ips": OPEResult(
            estimator="IPS",
            point_estimate=ips_point,
            ci_95_low=ips_low,
            ci_95_high=ips_high,
            baseline_behavior_return=baseline_return,
            n_episodes=len(test_trajectories),
            assumptions=ASSUMPTIONS,
            failure_cases=FAILURE_CASES,
            diagnostics=diagnostics,
        ),
        "dr": OPEResult(
            estimator="DoublyRobust",
            point_estimate=dr_point,
            ci_95_low=dr_low,
            ci_95_high=dr_high,
            baseline_behavior_return=baseline_return,
            n_episodes=len(test_trajectories),
            assumptions=ASSUMPTIONS,
            failure_cases=FAILURE_CASES,
            diagnostics=diagnostics,
        ),
    }
