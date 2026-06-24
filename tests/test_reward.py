"""Tests for reward construction and trajectory building."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.action_infer import infer_action_at_bot_turn
from src.models import Action, Call, Speaker, Turn
from src.parser import parse_transcript_dir, parse_transcript_file
from src.reward import RewardConfig, compute_terminal_reward, compute_turn_reward
from src.trajectory import build_trajectory_for_call, build_trajectories, summarise_trajectories

TRANSCRIPTS_DIR = Path("data/transcripts")


def _turn(index: int, speaker: Speaker, text: str, ts: str) -> Turn:
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S UTC")
    return Turn(index=index, speaker=speaker, text=text, timestamp=dt, timestamp_raw=ts)


class TestTurnReward:
    def test_collision_penalty(self) -> None:
        call = Call(
            call_sid="collision",
            source_path=Path("x.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hello", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "Okay", "2026-05-21 10:00:05 UTC"),
                _turn(
                    2,
                    Speaker.AI_ASSISTANT,
                    "And what's your budSure - ten percent to book.",
                    "2026-05-21 10:00:10 UTC",
                ),
            ],
        )
        reward = compute_turn_reward(call, 2, Action.PROVIDE_INFO)
        assert reward.components["collision"] == -0.30

    def test_busy_ignored_penalty(self) -> None:
        call = Call(
            call_sid="busy",
            source_path=Path("x.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hello", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "अभी टाइम नहीं है।", "2026-05-21 10:00:05 UTC"),
                _turn(2, Speaker.AI_ASSISTANT, "And what's your budget range?", "2026-05-21 10:00:10 UTC"),
            ],
        )
        reward = compute_turn_reward(call, 2, Action.ASK_BUDGET)
        assert reward.components["busy_ignored"] == -0.40

    def test_objection_handling_reward(self) -> None:
        call = Call(
            call_sid="objection",
            source_path=Path("x.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hello", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "How are you different?", "2026-05-21 10:00:05 UTC"),
                _turn(
                    2,
                    Speaker.AI_ASSISTANT,
                    "Fair point, I get that a lot - totally understand the skepticism.",
                    "2026-05-21 10:00:10 UTC",
                ),
            ],
        )
        reward = compute_turn_reward(call, 2, Action.HANDLE_OBJECTION)
        assert reward.components["handle_objection"] == 0.20

    def test_customer_substance_reward(self) -> None:
        call = Call(
            call_sid="substance",
            source_path=Path("x.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Which location?", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "Sector 150 Noida", "2026-05-21 10:00:05 UTC"),
            ],
        )
        reward = compute_turn_reward(call, 0, Action.ASK_LOCATION)
        assert reward.components["customer_substance"] == 0.25


class TestTerminalReward:
    def test_premature_exit_on_engaged_call(self) -> None:
        call = Call(
            call_sid="premature",
            source_path=Path("x.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hi - good time?", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "Yes, go on.", "2026-05-21 10:00:05 UTC"),
                _turn(2, Speaker.AI_ASSISTANT, "What type of property?", "2026-05-21 10:00:10 UTC"),
                _turn(3, Speaker.CUSTOMER, "Tell me more.", "2026-05-21 10:00:15 UTC"),
                _turn(4, Speaker.AI_ASSISTANT, "No problem. Have a nice day.", "2026-05-21 10:00:20 UTC"),
            ],
        )
        terminal = compute_terminal_reward(call)
        assert terminal.components["premature_exit"] == -0.80

    def test_hostile_end(self) -> None:
        call = Call(
            call_sid="hostile",
            source_path=Path("x.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hi", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "You are foolish.", "2026-05-21 10:00:05 UTC"),
                _turn(2, Speaker.AI_ASSISTANT, "Have a nice day.", "2026-05-21 10:00:10 UTC"),
            ],
        )
        terminal = compute_terminal_reward(call)
        assert terminal.components["hostile_end"] == -0.60


@pytest.mark.skipif(not TRANSCRIPTS_DIR.is_dir(), reason="dataset not present")
class TestTrajectoryBuilding:
    def test_build_single_call(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "0a48de35-5722-f494-6991-e2dc09b67c76.txt")
        trajectory = build_trajectory_for_call(call)
        assert len(trajectory.transitions) == len(call.ai_turns)
        assert trajectory.transitions[-1].done is True
        assert trajectory.total_return != 0.0

    def test_busy_call_has_negative_components(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "0a48de35-5722-f494-6991-e2dc09b67c76.txt")
        trajectory = build_trajectory_for_call(call)
        all_components = [
            name
            for transition in trajectory.transitions
            for name in transition.reward_breakdown
        ]
        assert "busy_ignored" in all_components or "collision" in all_components

    def test_objection_call_premature_or_negative_terminal(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "2e2fabba-cac4-5422-2f7b-85917c5528b8.txt")
        trajectory = build_trajectory_for_call(call)
        last = trajectory.transitions[-1]
        terminal_keys = set(last.reward_breakdown)
        assert terminal_keys & {
            "premature_exit",
            "wasted_engaged",
            "hostile_end",
            "neutral_end",
            "success",
        }

    def test_build_all_calls(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        trajectories = build_trajectories(calls)
        summary = summarise_trajectories(trajectories)
        assert summary.total_calls == 1500
        assert summary.total_transitions > 10_000
        assert summary.min_return < 0.0
        assert summary.max_return > 0.0

    def test_reward_breakdown_is_populated(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "2e2fabba-cac4-5422-2f7b-85917c5528b8.txt")
        trajectory = build_trajectory_for_call(call)
        assert trajectory.transitions[-1].reward_breakdown
        assert any(transition.reward_breakdown for transition in trajectory.transitions)
