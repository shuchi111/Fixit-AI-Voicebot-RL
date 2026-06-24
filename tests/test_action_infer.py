"""Tests for bot action inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.action_infer import (
    infer_action_at_bot_turn,
    infer_action_from_text,
    infer_actions_for_call,
)
from src.models import Action, Call, Speaker, Turn
from src.parser import parse_transcript_dir, parse_transcript_file

TRANSCRIPTS_DIR = Path("data/transcripts")
GOLD_LABELS_PATH = Path("tests/fixtures/action_gold_labels.json")


class TestInferActionFromText:
    def test_graceful_exit(self) -> None:
        inference = infer_action_from_text("No problem at all. Have a nice day.")
        assert inference.action is Action.GRACEFUL_EXIT

    def test_ask_budget(self) -> None:
        inference = infer_action_from_text("And what's your budget range for this?")
        assert inference.action is Action.ASK_BUDGET

    def test_handle_objection(self) -> None:
        inference = infer_action_from_text(
            "Fair point, I get that a lot - totally understand the skepticism."
        )
        assert inference.action is Action.HANDLE_OBJECTION

    def test_provide_info_on_collision(self) -> None:
        inference = infer_action_from_text(
            "And what's your budSure - it's ten percent to book, one percent a month."
        )
        assert inference.action is Action.PROVIDE_INFO

    def test_collision_location(self) -> None:
        inference = infer_action_from_text(
            "And what's your budWhich location are you interested in?"
        )
        assert inference.action is Action.ASK_LOCATION

    def test_collision_objection_beats_timeline(self) -> None:
        inference = infer_action_from_text(
            "What's your timeFair point, I get that a lot - totally understand the skepticism."
        )
        assert inference.action is Action.HANDLE_OBJECTION

    def test_busy_context_mismatch_on_budget(self) -> None:
        inference = infer_action_from_text(
            "And what's your budget range for this?",
            last_customer_text="अभी टाइम नहीं है।",
        )
        assert inference.action is Action.ASK_BUDGET
        assert inference.context_mismatch is True

    def test_greet_opener(self) -> None:
        inference = infer_action_from_text("Hello! This is Arya. Do you have a quick minute?")
        assert inference.action is Action.GREET


class TestGoldLabelAccuracy:
    @pytest.fixture(scope="class")
    def gold_labels(self) -> list[dict]:
        return json.loads(GOLD_LABELS_PATH.read_text(encoding="utf-8"))

    def test_gold_label_count(self, gold_labels: list[dict]) -> None:
        assert len(gold_labels) == 50

    def test_accuracy_on_gold_labels(self, gold_labels: list[dict]) -> None:
        correct = 0
        errors: list[str] = []
        for item in gold_labels:
            inference = infer_action_from_text(
                item["bot_text"],
                last_customer_text=item["last_customer_text"],
            )
            expected = Action(item["expected"])
            if inference.action is expected:
                correct += 1
            else:
                errors.append(
                    f"{item['bot_text'][:50]!r} -> {inference.action.value} "
                    f"(expected {expected.value}, rule={inference.rule})"
                )
        accuracy = correct / len(gold_labels)
        assert accuracy >= 0.80, (
            f"Accuracy {accuracy:.1%} below 80%. Errors:\n" + "\n".join(errors[:10])
        )


@pytest.mark.skipif(not TRANSCRIPTS_DIR.is_dir(), reason="dataset not present")
class TestRealTranscriptActions:
    def test_infer_all_calls(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        total = 0
        for call in calls:
            inferences = infer_actions_for_call(call)
            total += len(inferences)
            assert len(inferences) == len(call.ai_turns)
            for inference in inferences:
                assert isinstance(inference.action, Action)
        assert total > 10_000

    def test_known_call_actions(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "0a48de35-5722-f494-6991-e2dc09b67c76.txt")
        actions = [inf.action for inf in infer_actions_for_call(call)]
        assert Action.GREET in actions
        assert Action.GRACEFUL_EXIT in actions
        assert Action.ASK_BUDGET in actions

    def test_objection_call_actions(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "2e2fabba-cac4-5422-2f7b-85917c5528b8.txt")
        actions = [inf.action for inf in infer_actions_for_call(call)]
        assert actions.count(Action.HANDLE_OBJECTION) >= 2
        assert Action.GRACEFUL_EXIT in actions

    def test_context_mismatch_rate_is_nonzero(self) -> None:
        calls, _ = parse_transcript_dir(TRANSCRIPTS_DIR)
        mismatches = 0
        total = 0
        for call in calls:
            for inference in infer_actions_for_call(call):
                total += 1
                if inference.context_mismatch:
                    mismatches += 1
        assert mismatches > 0
        assert mismatches / total < 0.5

    def test_infer_action_at_bot_turn(self) -> None:
        call = parse_transcript_file(TRANSCRIPTS_DIR / "ba1828dc-fc95-db80-cb36-b07e99a474de.txt")
        bot_indices = [
            index
            for index, turn in enumerate(call.turns)
            if turn.speaker is Speaker.AI_ASSISTANT
        ]
        inference = infer_action_at_bot_turn(call, bot_indices[0])
        assert inference.action is Action.GREET
