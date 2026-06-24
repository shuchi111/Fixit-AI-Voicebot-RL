"""Tests for state feature extraction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.features import (
    FEATURE_NAMES,
    extract_state_at_bot_turn,
    extract_states_for_call,
    has_collision,
    hindi_ratio,
    score_presence,
)
from src.models import Call, Speaker, Turn
from src.parser import parse_transcript_dir, parse_transcript_file

TRANSCRIPTS_DIR = Path("data/transcripts")


def _turn(
    index: int,
    speaker: Speaker,
    text: str,
    ts: str,
) -> Turn:
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S UTC")
    return Turn(index=index, speaker=speaker, text=text, timestamp=dt, timestamp_raw=ts)


class TestFeatureHelpers:
    def test_hindi_ratio(self) -> None:
        assert hindi_ratio("अभी टाइम नहीं है।") > 0.3
        assert hindi_ratio("Hello there") == 0.0

    def test_collision_detection(self) -> None:
        assert has_collision("And what's your budSure - it's ten percent")
        assert not has_collision("What type of property are you looking for?")

    def test_engagement_score(self) -> None:
        assert score_presence("Yes, go on.", ("yes", "go on")) == 1.0
        assert score_presence("Everyone sells the same thing.", ("yes", "go on")) == 0.0


class TestExtractStateAtBotTurn:
    def test_busy_customer_before_budget_question(self) -> None:
        call = Call(
            call_sid="test-busy",
            source_path=Path("test.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hello! Do you have a minute?", "2026-05-28 15:19:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "अभी टाइम नहीं है।", "2026-05-28 15:19:13 UTC"),
                _turn(2, Speaker.AI_ASSISTANT, "And what's your budget range?", "2026-05-28 15:19:16 UTC"),
            ],
        )
        state = extract_state_at_bot_turn(call, bot_turn_index=2)
        assert state.features["busy_score"] == 1.0
        assert state.features["hindi_ratio"] > 0.0

    def test_barge_in_flag_when_bot_timestamp_earlier(self) -> None:
        call = Call(
            call_sid="test-barge",
            source_path=Path("test.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hi there", "2026-05-21 07:45:07 UTC"),
                _turn(1, Speaker.CUSTOMER, "You are foolish.", "2026-05-21 07:45:09 UTC"),
                _turn(2, Speaker.AI_ASSISTANT, "Have a nice day.", "2026-05-21 07:45:07 UTC"),
            ],
        )
        state = extract_state_at_bot_turn(call, bot_turn_index=2)
        assert state.features["barge_in_flag"] == 1.0
        assert state.features["hostility_score"] == 1.0

    def test_slot_filling_from_customer_history(self) -> None:
        call = Call(
            call_sid="test-slots",
            source_path=Path("test.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hi", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "Looking for apartment in Noida, 80 lakh", "2026-05-21 10:00:05 UTC"),
                _turn(2, Speaker.AI_ASSISTANT, "What's your timeline?", "2026-05-21 10:00:10 UTC"),
            ],
        )
        state = extract_state_at_bot_turn(call, bot_turn_index=2)
        assert state.features["property_filled"] == 1.0
        assert state.features["budget_filled"] == 1.0
        assert state.features["location_filled"] == 1.0
        assert state.features["slots_filled_count"] == 3.0

    def test_all_feature_names_present(self) -> None:
        call = Call(
            call_sid="test-all",
            source_path=Path("test.txt"),
            turns=[
                _turn(0, Speaker.AI_ASSISTANT, "Hello", "2026-05-21 10:00:00 UTC"),
                _turn(1, Speaker.CUSTOMER, "Okay", "2026-05-21 10:00:05 UTC"),
            ],
        )
        state = extract_state_at_bot_turn(call, bot_turn_index=0)
        assert list(state.features.keys()) == FEATURE_NAMES
        assert state.to_array(FEATURE_NAMES) == [state.features[name] for name in FEATURE_NAMES]

    def test_non_bot_turn_raises(self) -> None:
        call = Call(
            call_sid="test",
            source_path=Path("test.txt"),
            turns=[_turn(0, Speaker.CUSTOMER, "Hi", "2026-05-21 10:00:00 UTC")],
        )
        with pytest.raises(ValueError, match="not an AI ASSISTANT"):
            extract_state_at_bot_turn(call, 0)


@pytest.mark.skipif(not TRANSCRIPTS_DIR.is_dir(), reason="dataset not present")
class TestRealTranscriptFeatures:
    def test_known_call_busy_and_collision(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "0a48de35-5722-f494-6991-e2dc09b67c76.txt")
        states = extract_states_for_call(call)
        assert len(states) == len(call.ai_turns)
        busy_states = [s for s in states if s.features["busy_score"] == 1.0]
        assert busy_states
        collision_states = [s for s in states if s.features["bot_collision_last"] == 1.0]
        assert collision_states

    def test_objection_call_has_objection_features(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "2e2fabba-cac4-5422-2f7b-85917c5528b8.txt")
        states = extract_states_for_call(call)
        assert any(s.features["objection_score"] == 1.0 for s in states)
        assert any(s.features["engagement_score"] == 1.0 for s in states)

    def test_hostile_call(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "ba1828dc-fc95-db80-cb36-b07e99a474de.txt")
        states = extract_states_for_call(call)
        assert any(s.features["hostility_score"] == 1.0 for s in states)

    def test_all_calls_produce_valid_features(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        total_states = 0
        for call in calls:
            states = extract_states_for_call(call)
            total_states += len(states)
            for state in states:
                assert len(state.features) == len(FEATURE_NAMES)
                for name in FEATURE_NAMES:
                    value = state.features[name]
                    assert value == value  # not NaN
                    assert abs(value) != float("inf")
        assert total_states > 10_000
