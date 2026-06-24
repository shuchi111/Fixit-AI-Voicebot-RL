"""Core data models for the voicebot RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class Speaker(str, Enum):
    """Conversation participant."""

    AI_ASSISTANT = "AI ASSISTANT"
    CUSTOMER = "CUSTOMER"


class Action(str, Enum):
    """Abstract turn-level decision types (filled in by action_infer.py)."""

    GREET = "GREET"
    ASK_PROPERTY_TYPE = "ASK_PROPERTY_TYPE"
    ASK_BUDGET = "ASK_BUDGET"
    ASK_LOCATION = "ASK_LOCATION"
    ASK_TIMELINE = "ASK_TIMELINE"
    HANDLE_OBJECTION = "HANDLE_OBJECTION"
    PROVIDE_INFO = "PROVIDE_INFO"
    CLARIFY_REPEAT = "CLARIFY_REPEAT"
    ACK_BUSY_DEFER = "ACK_BUSY_DEFER"
    GRACEFUL_EXIT = "GRACEFUL_EXIT"


@dataclass(frozen=True)
class Turn:
    """A single utterance in a call transcript."""

    index: int
    speaker: Speaker
    text: str
    timestamp: datetime
    timestamp_raw: str


@dataclass
class Call:
    """Parsed representation of one phone call."""

    call_sid: str
    source_path: Path
    turns: list[Turn]
    declared_message_count: int | None = None
    message_count_mismatch: bool = False

    @property
    def message_count(self) -> int:
        return len(self.turns)

    @property
    def ai_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker is Speaker.AI_ASSISTANT]

    @property
    def customer_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker is Speaker.CUSTOMER]


@dataclass
class StateVector:
    """Numeric state at a bot decision point."""

    features: dict[str, float]
    turn_index: int
    call_sid: str

    def to_array(self, feature_names: list[str]) -> list[float]:
        return [self.features.get(name, 0.0) for name in feature_names]


@dataclass
class Transition:
    """One MDP transition at a bot decision point."""

    call_sid: str
    turn_index: int
    state: StateVector
    action: Action | None
    reward: float
    next_state: StateVector | None
    done: bool
    reward_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Sequence of transitions for one call episode."""

    call_sid: str
    transitions: list[Transition]

    @property
    def total_return(self) -> float:
        return sum(t.reward for t in self.transitions)


@dataclass
class ParseSummary:
    """Aggregate statistics from parsing a transcript directory."""

    total_files: int
    parsed_calls: int
    failed_files: list[str]
    message_counts: list[int]
    mismatch_count: int

    def to_dict(self) -> dict[str, Any]:
        counts = self.message_counts
        return {
            "total_files": self.total_files,
            "parsed_calls": self.parsed_calls,
            "failed_files": self.failed_files,
            "mismatch_count": self.mismatch_count,
            "message_count_min": min(counts) if counts else 0,
            "message_count_max": max(counts) if counts else 0,
            "message_count_mean": sum(counts) / len(counts) if counts else 0.0,
        }
