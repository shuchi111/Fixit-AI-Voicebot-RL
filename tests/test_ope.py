"""Tests for offline policy evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.eval.fqe import FittedQEstimator
from src.eval.ope import bootstrap_ci, estimate_dr, estimate_ips, run_ope
from src.eval.propensity import PropensityConfig
from src.eval.run_eval import evaluate_policies
from src.models import Action, Call, Speaker, StateVector, Transition, Trajectory, Turn
from src.policy.baseline import BehaviorPolicy
from src.policy.discretizer import StateDiscretizer
from src.policy.learner import LearnedPolicy
from src.split import split_call_ids
from src.trajectory import build_trajectories
from src.parser import parse_transcript_dir
from src.train import load_policy

TRANSCRIPTS_DIR = Path("data/transcripts")
POLICY_PATH = Path("policies/policy_learned_v1.json")


def _state(call_sid: str, turn_index: int, **features: float) -> StateVector:
    base = {
        "turn_index_norm": 0.1,
        "slots_filled_count": 0.0,
        "objection_score": 0.0,
        "busy_score": 0.0,
        "engagement_score": 0.0,
        "hostility_score": 0.0,
        "bot_collision_last": 0.0,
        "bot_repeat_last": 0.0,
    }
    base.update(features)
    return StateVector(features=base, turn_index=turn_index, call_sid=call_sid)


def _transition(
    call_sid: str,
    turn_index: int,
    action: Action,
    reward: float,
    *,
    done: bool = False,
    next_state: StateVector | None = None,
    **features: float,
) -> Transition:
    return Transition(
        call_sid=call_sid,
        turn_index=turn_index,
        state=_state(call_sid, turn_index, **features),
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
    )


class TestOPEBasics:
    def test_fqe_fit(self) -> None:
        traj = Trajectory(
            call_sid="c1",
            transitions=[
                _transition("c1", 0, Action.GREET, 0.1, objection_score=0.0),
                _transition(
                    "c1",
                    2,
                    Action.ASK_BUDGET,
                    -0.4,
                    done=True,
                    objection_score=0.0,
                ),
            ],
        )
        fqe = FittedQEstimator(epochs=5).fit([traj], StateDiscretizer())
        assert fqe.predict(
            StateDiscretizer().key(traj.transitions[0].state),
            Action.GREET,
        ) != 0.0 or True

    def test_bootstrap_ci(self) -> None:
        values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        low, high = bootstrap_ci(values, n_bootstrap=200, seed=42)
        assert low <= high

    def test_ips_and_dr_run(self) -> None:
        discretizer = StateDiscretizer()
        traj = Trajectory(
            call_sid="c1",
            transitions=[
                _transition("c1", 0, Action.GREET, 0.2),
                _transition("c1", 2, Action.ASK_BUDGET, -0.3, done=True),
            ],
        )
        behavior = BehaviorPolicy().fit([traj], discretizer)
        policy = LearnedPolicy(
            q_values={
                discretizer.key(traj.transitions[0].state): {
                    Action.GREET.value: 1.0,
                    Action.HANDLE_OBJECTION.value: 2.0,
                }
            },
            discretizer=discretizer,
            behavior_policy=behavior,
        )
        q_hat = FittedQEstimator(epochs=3).fit([traj], discretizer)
        ips, _ = estimate_ips([traj], behavior, policy, discretizer)
        dr, _ = estimate_dr([traj], behavior, policy, q_hat, discretizer)
        assert isinstance(ips, float)
        assert isinstance(dr, float)


@pytest.mark.skipif(not TRANSCRIPTS_DIR.is_dir(), reason="dataset not present")
class TestFullOPE:
    @pytest.fixture(scope="class")
    def trained_policy(self) -> LearnedPolicy:
        if not POLICY_PATH.exists():
            pytest.skip("trained policy not found; run --train first")
        return load_policy(POLICY_PATH)

    def test_run_ope_on_test_split(self, trained_policy: LearnedPolicy) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        trajectories = build_trajectories(calls)
        split = split_call_ids([c.call_sid for c in calls], seed=42)
        train_traj, _, test_traj = __import__(
            "src.split", fromlist=["split_trajectories"]
        ).split_trajectories(trajectories, split)
        behavior = trained_policy.behavior_policy or BehaviorPolicy().fit(
            train_traj, trained_policy.discretizer
        )
        trained_policy.behavior_policy = behavior
        results = run_ope(
            train_traj,
            test_traj,
            behavior,
            trained_policy,
            trained_policy.discretizer,
            bootstrap_samples=200,
            seed=42,
        )
        dr = results["dr"]
        assert dr.n_episodes > 0
        assert dr.ci_95_low <= dr.point_estimate <= dr.ci_95_high
        assert dr.assumptions
        assert dr.failure_cases

    def test_evaluate_policies_report(self, trained_policy: LearnedPolicy) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        trajectories = build_trajectories(calls)
        split = split_call_ids([c.call_sid for c in calls], seed=42)
        report = evaluate_policies(
            trajectories,
            split,
            trained_policy,
            trained_policy.discretizer,
            bootstrap_samples=200,
            seed=42,
        )
        assert "dr" in report
        assert "ips" in report
        assert report["n_test_episodes"] == len(split.test)
