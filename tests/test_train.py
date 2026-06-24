"""Tests for policy training and replay evaluation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.features import extract_state_at_bot_turn
from src.models import Action, Call, Speaker, StateVector, Transition, Trajectory, Turn
from src.policy.baseline import BehaviorPolicy
from src.policy.discretizer import StateDiscretizer
from src.policy.constraints import PolicyConstraints
from src.policy.learner import QLearningConfig, train_q_learning
from src.eval.replay import (
    build_reward_lookup,
    evaluate_behavior_returns,
    evaluate_learned_policy,
)
from src.split import split_call_ids
from src.train import train_policy
from src.trajectory import build_trajectories
from src.parser import parse_transcript_dir

TRANSCRIPTS_DIR = Path("data/transcripts")


def _turn(index: int, speaker: Speaker, text: str, ts: str) -> Turn:
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S UTC")
    return Turn(index=index, speaker=speaker, text=text, timestamp=dt, timestamp_raw=ts)


def _make_call(call_sid: str, customer_busy: bool = False) -> Call:
    customer = "अभी टाइम नहीं है।" if customer_busy else "Yes, go on."
    return Call(
        call_sid=call_sid,
        source_path=Path("x.txt"),
        turns=[
            _turn(0, Speaker.AI_ASSISTANT, "Hello! Good time?", "2026-05-21 10:00:00 UTC"),
            _turn(1, Speaker.CUSTOMER, customer, "2026-05-21 10:00:05 UTC"),
            _turn(2, Speaker.AI_ASSISTANT, "What type of property?", "2026-05-21 10:00:10 UTC"),
            _turn(3, Speaker.CUSTOMER, "Apartment in Noida", "2026-05-21 10:00:15 UTC"),
            _turn(4, Speaker.AI_ASSISTANT, "What's your budget?", "2026-05-21 10:00:20 UTC"),
            _turn(5, Speaker.CUSTOMER, "80 lakh", "2026-05-21 10:00:25 UTC"),
            _turn(6, Speaker.AI_ASSISTANT, "Which location?", "2026-05-21 10:00:30 UTC"),
            _turn(7, Speaker.CUSTOMER, "Sector 150", "2026-05-21 10:00:35 UTC"),
            _turn(8, Speaker.AI_ASSISTANT, "Timeline for buying?", "2026-05-21 10:00:40 UTC"),
            _turn(9, Speaker.CUSTOMER, "Six months", "2026-05-21 10:00:45 UTC"),
            _turn(10, Speaker.AI_ASSISTANT, "Great, thanks!", "2026-05-21 10:00:50 UTC"),
        ],
    )


class TestDiscretizerAndBaseline:
    def test_discretizer_key_is_stable(self) -> None:
        call = _make_call("d1")
        state = extract_state_at_bot_turn(call, 2)
        discretizer = StateDiscretizer()
        assert discretizer.key(state) == discretizer.key(state)

    def test_behavior_policy_fit(self) -> None:
        call = _make_call("d2")
        traj = build_trajectories([call])[0]
        discretizer = StateDiscretizer()
        policy = BehaviorPolicy().fit([traj], discretizer)
        assert policy.action_counts


class TestQLearning:
    def test_q_learning_runs(self) -> None:
        calls = [_make_call(f"c{i}") for i in range(10)]
        trajectories = build_trajectories(calls)
        discretizer = StateDiscretizer()
        behavior = BehaviorPolicy().fit(trajectories, discretizer)
        policy = train_q_learning(
            trajectories,
            discretizer=discretizer,
            behavior_policy=behavior,
            config=QLearningConfig(epochs=5, alpha=0.2, gamma=0.9),
        )
        assert policy.q_values


@pytest.mark.skipif(not TRANSCRIPTS_DIR.is_dir(), reason="dataset not present")
class TestFullTraining:
    def test_learned_beats_baseline_on_val(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        trajectories = build_trajectories(calls)
        split = split_call_ids([c.call_sid for c in calls], seed=42)
        result = train_policy(
            trajectories,
            split,
            discretizer=StateDiscretizer(),
            q_config=QLearningConfig(epochs=40, alpha=0.12, gamma=0.95),
            constraints=PolicyConstraints(
                max_kl_divergence=0.8,
                min_q_improvement=0.12,
            ),
        )
        assert result.learned_val_return >= result.baseline_val_return
        assert len(result.learning_curve) == 40

    def test_replay_evaluation_helpers(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        trajectories = build_trajectories(calls[:100])
        split = split_call_ids([c.call_sid for c in calls[:100]], seed=42)
        train_traj, val_traj, _ = __import__(
            "src.split", fromlist=["split_trajectories"]
        ).split_trajectories(trajectories, split)
        discretizer = StateDiscretizer()
        lookup = build_reward_lookup(train_traj, discretizer)
        baseline = evaluate_behavior_returns(val_traj)
        assert baseline.num_episodes > 0
        assert lookup
