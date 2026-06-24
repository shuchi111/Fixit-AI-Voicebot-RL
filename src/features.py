"""Extract interpretable state features at each bot decision point."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.models import Call, Speaker, StateVector, Turn

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
COLLISION_RE = re.compile(r"[a-z][A-Z]")
NUMBER_RE = re.compile(r"\d+")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

ENGAGEMENT_PATTERNS = (
    "yes",
    "go on",
    "tell me",
    "okay",
    "ok",
    "sure",
    "हाँ",
    "हां",
    "ji",
    "जी",
    "continue",
)
BUSY_PATTERNS = (
    "time nahi",
    "टाइम नहीं",
    "busy",
    "call later",
    "call back",
    "not now",
    "bad time",
    "अभी टाइम",
    "बाद में",
)
HOSTILITY_PATTERNS = (
    "foolish",
    "bakwas",
    "बकवास",
    "stop calling",
    "don't call",
    "do not call",
    "idiot",
    "annoying",
    "harass",
)
OBJECTION_PATTERNS = (
    "how are you different",
    "why should i",
    "everyone sells",
    "same thing",
    "skeptic",
    "scam",
    "trust",
    "different from",
)
CORRECTION_PATTERNS = (
    "no i said",
    "i meant",
    "not that",
    "i said",
    "wrong",
)
SUBSTANCE_PATTERNS = (
    "apartment",
    "villa",
    "flat",
    "bhk",
    "lakh",
    "crore",
    "noida",
    "gurgaon",
    "gurugram",
    "mumbai",
    "pune",
    "sector",
)
PROPERTY_PATTERNS = ("apartment", "villa", "flat", "bhk", "property")
BUDGET_PATTERNS = ("lakh", "crore", "budget", "rupee", "rs", "₹")
LOCATION_PATTERNS = (
    "noida",
    "gurgaon",
    "gurugram",
    "mumbai",
    "pune",
    "delhi",
    "sector",
    "location",
    "where",
)
TIMELINE_PATTERNS = (
    "month",
    "year",
    "soon",
    "timeline",
    "buying",
    "ready to",
    "possession",
    "next quarter",
)

FEATURE_NAMES: list[str] = [
    "turn_index_norm",
    "slots_filled_count",
    "property_filled",
    "budget_filled",
    "location_filled",
    "timeline_filled",
    "cust_utt_length",
    "hindi_ratio",
    "engagement_score",
    "busy_score",
    "hostility_score",
    "objection_score",
    "correction_score",
    "substance_score",
    "short_reply",
    "repeat_customer_utt",
    "silence_gap_sec",
    "barge_in_flag",
    "bot_collision_last",
    "bot_repeat_last",
    "consecutive_objections",
    "cust_still_engaged",
    "opener_type",
    "call_hour_bucket",
]


@dataclass(frozen=True)
class SlotStatus:
    property_filled: bool
    budget_filled: bool
    location_filled: bool
    timeline_filled: bool

    @property
    def filled_count(self) -> int:
        return sum(
            (
                self.property_filled,
                self.budget_filled,
                self.location_filled,
                self.timeline_filled,
            )
        )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = _normalise(text)
    return any(pattern in lowered for pattern in patterns)


def hindi_ratio(text: str) -> float:
    if not text:
        return 0.0
    devanagari = len(DEVANAGARI_RE.findall(text))
    return devanagari / max(len(text), 1)


def score_presence(text: str, patterns: tuple[str, ...]) -> float:
    return 1.0 if _contains_any(text, patterns) else 0.0


def is_short_reply(text: str, max_words: int = 3) -> float:
    words = WORD_RE.findall(text)
    return 1.0 if 0 < len(words) <= max_words else 0.0


def has_collision(text: str) -> bool:
    return bool(COLLISION_RE.search(text))


def _normalise_question(text: str) -> str:
    return _normalise(text)[:60]


def is_repeat_question(current: str, previous_questions: list[str]) -> bool:
    current_norm = _normalise_question(current)
    return any(current_norm == _normalise_question(prev) for prev in previous_questions)


def infer_slots_from_text(text: str) -> SlotStatus:
    lowered = _normalise(text)
    has_number = bool(NUMBER_RE.search(lowered))
    return SlotStatus(
        property_filled=_contains_any(lowered, PROPERTY_PATTERNS),
        budget_filled=_contains_any(lowered, BUDGET_PATTERNS) or has_number,
        location_filled=_contains_any(lowered, LOCATION_PATTERNS),
        timeline_filled=_contains_any(lowered, TIMELINE_PATTERNS),
    )


def merge_slots(existing: SlotStatus, new: SlotStatus) -> SlotStatus:
    return SlotStatus(
        property_filled=existing.property_filled or new.property_filled,
        budget_filled=existing.budget_filled or new.budget_filled,
        location_filled=existing.location_filled or new.location_filled,
        timeline_filled=existing.timeline_filled or new.timeline_filled,
    )


def accumulate_slots(turns: list[Turn]) -> SlotStatus:
    slots = SlotStatus(False, False, False, False)
    for turn in turns:
        slots = merge_slots(slots, infer_slots_from_text(turn.text))
    return slots


def opener_type_bucket(text: str) -> float:
    lowered = _normalise(text)
    if "arya" in lowered:
        return 0.0
    if "rohan" in lowered:
        return 1.0
    if "good time" in lowered or "quick minute" in lowered:
        return 2.0
    return 3.0


def hour_bucket(timestamp_hour: int) -> float:
    if 5 <= timestamp_hour < 12:
        return 0.0
    if 12 <= timestamp_hour < 17:
        return 1.0
    return 2.0


def _previous_bot_turns(turns: list[Turn], before_index: int) -> list[Turn]:
    return [t for t in turns[:before_index] if t.speaker is Speaker.AI_ASSISTANT]


def _recent_customer_turns(turns: list[Turn], before_index: int, limit: int) -> list[Turn]:
    customers = [t for t in turns[:before_index] if t.speaker is Speaker.CUSTOMER]
    return customers[-limit:]


def _utterance_text(value: Turn | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.text


def extract_customer_features(
    current: Turn | str | None,
    previous: Turn | str | None,
) -> dict[str, float]:
    if current is None:
        return {
            "cust_utt_length": 0.0,
            "hindi_ratio": 0.0,
            "engagement_score": 0.0,
            "busy_score": 0.0,
            "hostility_score": 0.0,
            "objection_score": 0.0,
            "correction_score": 0.0,
            "substance_score": 0.0,
            "short_reply": 0.0,
            "repeat_customer_utt": 0.0,
        }

    text = _utterance_text(current) or ""
    prev_text = _utterance_text(previous)
    return {
        "cust_utt_length": math.log1p(len(text)),
        "hindi_ratio": hindi_ratio(text),
        "engagement_score": score_presence(text, ENGAGEMENT_PATTERNS),
        "busy_score": score_presence(text, BUSY_PATTERNS),
        "hostility_score": score_presence(text, HOSTILITY_PATTERNS),
        "objection_score": score_presence(text, OBJECTION_PATTERNS),
        "correction_score": score_presence(text, CORRECTION_PATTERNS),
        "substance_score": score_presence(text, SUBSTANCE_PATTERNS),
        "short_reply": is_short_reply(text),
        "repeat_customer_utt": (
            1.0
            if prev_text is not None and _normalise(prev_text) == _normalise(text)
            else 0.0
        ),
    }


def extract_state_at_bot_turn(call: Call, bot_turn_index: int) -> StateVector:
    """Build state vector immediately before the bot speaks at `bot_turn_index`."""
    turns = call.turns
    bot_turn = turns[bot_turn_index]
    if bot_turn.speaker is not Speaker.AI_ASSISTANT:
        raise ValueError(f"Turn {bot_turn_index} is not an AI ASSISTANT turn")

    history = turns[:bot_turn_index]
    prior_bot_turns = _previous_bot_turns(turns, bot_turn_index)
    customer_turns_before = [t for t in history if t.speaker is Speaker.CUSTOMER]
    last_customer = customer_turns_before[-1] if customer_turns_before else None
    prev_customer = customer_turns_before[-2] if len(customer_turns_before) >= 2 else None

    slots = accumulate_slots(history)
    customer_features = extract_customer_features(last_customer, prev_customer)

    silence_gap_sec = 0.0
    barge_in_flag = 0.0
    if last_customer is not None:
        silence_gap_sec = max(
            0.0,
            (bot_turn.timestamp - last_customer.timestamp).total_seconds(),
        )
        barge_in_flag = 1.0 if bot_turn.timestamp < last_customer.timestamp else 0.0

    bot_collision_last = 0.0
    bot_repeat_last = 0.0
    if prior_bot_turns:
        last_bot = prior_bot_turns[-1]
        bot_collision_last = 1.0 if has_collision(last_bot.text) else 0.0
        recent_questions = [t.text for t in prior_bot_turns[-2:]]
        bot_repeat_last = 1.0 if is_repeat_question(last_bot.text, recent_questions[:-1]) else 0.0

    recent_customers = _recent_customer_turns(turns, bot_turn_index, limit=3)
    consecutive_objections = float(
        sum(1 for turn in recent_customers if _contains_any(turn.text, OBJECTION_PATTERNS))
    )
    cust_still_engaged = (
        1.0
        if customer_features["engagement_score"] > 0 and customer_features["hostility_score"] == 0
        else 0.0
    )

    first_turn = turns[0]
    opener = opener_type_bucket(first_turn.text)
    hour = hour_bucket(first_turn.timestamp.hour)

    features: dict[str, float] = {
        "turn_index_norm": bot_turn_index / max(len(turns) - 1, 1),
        "slots_filled_count": float(slots.filled_count),
        "property_filled": float(slots.property_filled),
        "budget_filled": float(slots.budget_filled),
        "location_filled": float(slots.location_filled),
        "timeline_filled": float(slots.timeline_filled),
        **customer_features,
        "silence_gap_sec": silence_gap_sec,
        "barge_in_flag": barge_in_flag,
        "bot_collision_last": bot_collision_last,
        "bot_repeat_last": bot_repeat_last,
        "consecutive_objections": consecutive_objections,
        "cust_still_engaged": cust_still_engaged,
        "opener_type": opener,
        "call_hour_bucket": hour,
    }

    validate_features(features)
    return StateVector(
        features=features,
        turn_index=bot_turn_index,
        call_sid=call.call_sid,
    )


def extract_states_for_call(call: Call) -> list[StateVector]:
    """Extract one state vector per bot turn in the call."""
    return [
        extract_state_at_bot_turn(call, index)
        for index, turn in enumerate(call.turns)
        if turn.speaker is Speaker.AI_ASSISTANT
    ]


def validate_features(features: dict[str, float]) -> None:
    for name in FEATURE_NAMES:
        value = features[name]
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Invalid feature value for {name}: {value}")
