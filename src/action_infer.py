"""Infer abstract turn-level actions from bot utterances."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.features import BUSY_PATTERNS, score_presence
from src.models import Action, Call, Speaker, Turn

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ActionInference:
    """Inferred action for one bot turn."""

    action: Action
    rule: str
    bot_text: str
    turn_index: int
    last_customer_text: str | None
    context_mismatch: bool


def _normalise(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip().lower())


def _customer_busy(customer_text: str | None) -> bool:
    if not customer_text:
        return False
    return score_presence(customer_text, BUSY_PATTERNS) > 0


def infer_action_from_text(
    bot_text: str,
    *,
    last_customer_text: str | None = None,
    turn_index: int = 0,
) -> ActionInference:
    """
    Map a bot utterance to an abstract action using a priority-ordered rule list.

    The classifier infers the action the bot *actually took* from its words.
    `context_mismatch` is True when that action ignored an obvious customer signal
    (e.g. customer said they were busy but the bot asked for budget).
    """
    bot = _normalise(bot_text)

    def result(action: Action, rule: str, mismatch: bool = False) -> ActionInference:
        expected_busy = _customer_busy(last_customer_text)
        busy_mismatch = expected_busy and action not in {
            Action.ACK_BUSY_DEFER,
            Action.GRACEFUL_EXIT,
        }
        return ActionInference(
            action=action,
            rule=rule,
            bot_text=bot_text,
            turn_index=turn_index,
            last_customer_text=last_customer_text,
            context_mismatch=busy_mismatch or mismatch,
        )

    if any(
        phrase in bot
        for phrase in ("have a nice day", "leave it here", "nice day", "goodbye", "bye")
    ):
        return result(Action.GRACEFUL_EXIT, "graceful_exit_phrase")

    if any(
        phrase in bot
        for phrase in (
            "call you back",
            "call back",
            "reach out later",
            "talk later",
            "another time",
        )
    ):
        return result(Action.ACK_BUSY_DEFER, "explicit_defer")

    if _customer_busy(last_customer_text) and "no problem" in bot:
        return result(Action.ACK_BUSY_DEFER, "busy_ack_no_problem")

    if any(
        phrase in bot
        for phrase in (
            "fair point",
            "fair concern",
            "skepticism",
            "how we're different",
            "how we are different",
            "understand the skepticism",
            "explain how",
        )
    ):
        return result(Action.HANDLE_OBJECTION, "objection_response")

    if any(
        phrase in bot
        for phrase in (
            "percent",
            "handover",
            "payment",
            "per month",
            "to book",
        )
    ):
        return result(Action.PROVIDE_INFO, "payment_or_plan_info")

    if any(
        phrase in bot
        for phrase in (
            "apartment or villa",
            "type of property",
            "property are you looking",
            "property type",
        )
    ):
        return result(Action.ASK_PROPERTY_TYPE, "ask_property_type")

    if "budget" in bot:
        return result(Action.ASK_BUDGET, "ask_budget")

    if any(
        phrase in bot
        for phrase in ("which location", "location are you", "where are you looking")
    ):
        return result(Action.ASK_LOCATION, "ask_location")

    if any(
        phrase in bot
        for phrase in ("timeline", "when are you planning", "when do you plan")
    ):
        return result(Action.ASK_TIMELINE, "ask_timeline")

    if "buying" in bot and "timeline" not in bot:
        return result(Action.ASK_TIMELINE, "ask_buying_timeline")

    if any(
        phrase in bot
        for phrase in (
            "good time to talk",
            "quick minute",
            "check in",
            "following up",
            "shown interest",
            "filled out a form",
            "earlier showed interest",
        )
    ) or bot.startswith("hello") or bot.startswith("hi "):
        return result(Action.GREET, "greeting_or_opener")

    if any(
        phrase in bot
        for phrase in ("repeat that", "say that again", "didn't catch", "pardon", "clearly")
    ):
        return result(Action.CLARIFY_REPEAT, "clarify_repeat")

    return result(Action.CLARIFY_REPEAT, "fallback_unmatched")


def infer_action_at_bot_turn(call: Call, bot_turn_index: int) -> ActionInference:
    """Infer action for a specific bot turn using conversational context."""
    turn = call.turns[bot_turn_index]
    if turn.speaker is not Speaker.AI_ASSISTANT:
        raise ValueError(f"Turn {bot_turn_index} is not an AI ASSISTANT turn")

    customer_turns = [
        t for t in call.turns[:bot_turn_index] if t.speaker is Speaker.CUSTOMER
    ]
    last_customer = customer_turns[-1].text if customer_turns else None

    return infer_action_from_text(
        turn.text,
        last_customer_text=last_customer,
        turn_index=bot_turn_index,
    )


def infer_actions_for_call(call: Call) -> list[ActionInference]:
    """Infer actions for every bot turn in a call."""
    return [
        infer_action_at_bot_turn(call, index)
        for index, turn in enumerate(call.turns)
        if turn.speaker is Speaker.AI_ASSISTANT
    ]


def action_distribution(inferences: list[ActionInference]) -> dict[str, int]:
    counts: dict[str, int] = {action.value: 0 for action in Action}
    for inference in inferences:
        counts[inference.action.value] += 1
    return counts
