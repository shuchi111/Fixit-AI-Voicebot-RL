"""Discretise continuous state features into interpretable cluster keys."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import StateVector


@dataclass(frozen=True)
class StateDiscretizer:
    """Bucket selected features into a compact string cluster id."""

    turn_bins: int = 6

    def key_from_features(self, features: dict[str, float]) -> str:
        slots = int(features.get("slots_filled_count", 0))
        objection = int(features.get("objection_score", 0) > 0)
        busy = int(features.get("busy_score", 0) > 0)
        engagement = int(features.get("engagement_score", 0) > 0)
        hostility = int(features.get("hostility_score", 0) > 0)
        collision = int(features.get("bot_collision_last", 0) > 0)
        repeat = int(features.get("bot_repeat_last", 0) > 0)
        turn_bucket = min(
            self.turn_bins - 1,
            int(features.get("turn_index_norm", 0.0) * self.turn_bins),
        )
        return (
            f"s{slots}|o{objection}|b{busy}|e{engagement}|h{hostility}"
            f"|c{collision}|r{repeat}|t{turn_bucket}"
        )

    def key(self, state: StateVector) -> str:
        return self.key_from_features(state.features)

    def to_dict(self) -> dict:
        return {"turn_bins": self.turn_bins}

    @classmethod
    def from_dict(cls, data: dict) -> StateDiscretizer:
        return cls(turn_bins=int(data.get("turn_bins", 6)))
