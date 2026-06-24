"""Construct turn-level and terminal rewards from conversation signals."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.action_infer import infer_action_at_bot_turn
from src.features import (
    ENGAGEMENT_PATTERNS,
    HOSTILITY_PATTERNS,
    accumulate_slots,
    extract_customer_features,
    extract_state_at_bot_turn,
    has_collision,
    score_presence,
)
from src.models import Action, Call, Speaker, Transition

ASK_ACTIONS = {
    Action.ASK_PROPERTY_TYPE,
    Action.ASK_BUDGET,
    Action.ASK_LOCATION,
    Action.ASK_TIMELINE,
}


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights loaded from config."""

    collision: float = -0.30
    repeat_question: float = -0.25
    objection_ignored: float = -0.50
    busy_ignored: float = -0.40
    customer_substance: float = 0.25
    handle_objection: float = 0.20
    handle_busy: float = 0.30
    customer_engagement: float = 0.10
    hostility_increase: float = -0.35
    success_terminal: float = 1.00
    good_deferral_terminal: float = 0.40
    premature_exit_terminal: float = -0.80
    hostile_end_terminal: float = -0.60
    wasted_engaged_terminal: float = -0.50
    min_success_turns: int = 8
    min_success_slots: int = 3
    min_engagement_signals: int = 3

    @classmethod
    def from_dict(cls, data: dict) -> RewardConfig:
        turn = data.get("turn", {})
        terminal = data.get("terminal", {})
        thresholds = data.get("thresholds", {})
        return cls(
            collision=float(turn.get("collision", -0.30)),
            repeat_question=float(turn.get("repeat_question", -0.25)),
            objection_ignored=float(turn.get("objection_ignored", -0.50)),
            busy_ignored=float(turn.get("busy_ignored", -0.40)),
            customer_substance=float(turn.get("customer_substance", 0.25)),
            handle_objection=float(turn.get("handle_objection", 0.20)),
            handle_busy=float(turn.get("handle_busy", 0.30)),
            customer_engagement=float(turn.get("customer_engagement", 0.10)),
            hostility_increase=float(turn.get("hostility_increase", -0.35)),
            success_terminal=float(terminal.get("success", 1.00)),
            good_deferral_terminal=float(terminal.get("good_deferral", 0.40)),
            premature_exit_terminal=float(terminal.get("premature_exit", -0.80)),
            hostile_end_terminal=float(terminal.get("hostile_end", -0.60)),
            wasted_engaged_terminal=float(terminal.get("wasted_engaged", -0.50)),
            min_success_turns=int(thresholds.get("min_success_turns", 8)),
            min_success_slots=int(thresholds.get("min_success_slots", 3)),
            min_engagement_signals=int(thresholds.get("min_engagement_signals", 3)),
        )


@dataclass
class RewardBreakdown:
    """Per-turn reward components for inspection."""

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        if value == 0.0:
            return
        self.components[name] = value
        self.total += value


def _next_customer_turn(call: Call, after_index: int) -> str | None:
    for turn in call.turns[after_index + 1 :]:
        if turn.speaker is Speaker.CUSTOMER:
            return turn.text
    return None


def _last_customer_before(call: Call, before_index: int) -> str | None:
    for turn in reversed(call.turns[:before_index]):
        if turn.speaker is Speaker.CUSTOMER:
            return turn.text
    return None


def _count_engagement_signals(call: Call) -> int:
    count = 0
    for turn in call.turns:
        if turn.speaker is Speaker.CUSTOMER:
            if score_presence(turn.text, ENGAGEMENT_PATTERNS) > 0:
                count += 1
    return count


def _call_had_busy_signal(call: Call) -> bool:
    from src.features import BUSY_PATTERNS

    return any(
        turn.speaker is Speaker.CUSTOMER and score_presence(turn.text, BUSY_PATTERNS) > 0
        for turn in call.turns
    )


def compute_turn_reward(
    call: Call,
    bot_turn_index: int,
    action: Action,
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute immediate + next-customer shaping reward for one bot turn."""
    cfg = config or RewardConfig()
    breakdown = RewardBreakdown()
    state = extract_state_at_bot_turn(call, bot_turn_index)
    features = state.features
    bot_turn = call.turns[bot_turn_index]

    if has_collision(bot_turn.text):
        breakdown.add("collision", cfg.collision)

    if features["bot_repeat_last"] > 0:
        breakdown.add("repeat_question", cfg.repeat_question)

    if features["objection_score"] > 0.5 and action in ASK_ACTIONS:
        breakdown.add("objection_ignored", cfg.objection_ignored)

    if features["busy_score"] > 0.5 and action not in {
        Action.ACK_BUSY_DEFER,
        Action.GRACEFUL_EXIT,
    }:
        breakdown.add("busy_ignored", cfg.busy_ignored)

    if features["objection_score"] > 0.5 and action is Action.HANDLE_OBJECTION:
        breakdown.add("handle_objection", cfg.handle_objection)

    if features["busy_score"] > 0.5 and action is Action.ACK_BUSY_DEFER:
        breakdown.add("handle_busy", cfg.handle_busy)

    next_customer_text = _next_customer_turn(call, bot_turn_index)
    if next_customer_text:
        next_features = extract_customer_features(next_customer_text, None)
        if next_features["substance_score"] > 0:
            breakdown.add("customer_substance", cfg.customer_substance)
        if next_features["engagement_score"] > 0:
            breakdown.add("customer_engagement", cfg.customer_engagement)

        prev_customer_text = _last_customer_before(call, bot_turn_index)
        prev_hostile = (
            score_presence(prev_customer_text or "", HOSTILITY_PATTERNS) > 0
        )
        next_hostile = score_presence(next_customer_text, HOSTILITY_PATTERNS) > 0
        if next_hostile and not prev_hostile:
            breakdown.add("hostility_increase", cfg.hostility_increase)

    return breakdown


def compute_terminal_reward(
    call: Call,
    *,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Compute sparse terminal reward at episode end."""
    cfg = config or RewardConfig()
    breakdown = RewardBreakdown()

    customer_turns = [t for t in call.turns if t.speaker is Speaker.CUSTOMER]
    slots = accumulate_slots(customer_turns)
    last_customer_text = _last_customer_before(call, len(call.turns))
    last_customer_features = extract_customer_features(last_customer_text, None)
    bot_actions = [infer_action_at_bot_turn(call, t.index).action for t in call.ai_turns]
    engagement_count = _count_engagement_signals(call)
    had_busy = _call_had_busy_signal(call)

    if (
        slots.filled_count >= cfg.min_success_slots
        and last_customer_features["hostility_score"] == 0
        and call.message_count >= cfg.min_success_turns
    ):
        breakdown.add("success", cfg.success_terminal)
        return breakdown

    if (
        had_busy
        and Action.ACK_BUSY_DEFER in bot_actions
        and bot_actions[-1] is Action.GRACEFUL_EXIT
    ):
        breakdown.add("good_deferral", cfg.good_deferral_terminal)
        return breakdown

    if (
        bot_actions
        and bot_actions[-1] is Action.GRACEFUL_EXIT
        and engagement_count >= 2
        and slots.filled_count < cfg.min_success_slots
    ):
        breakdown.add("premature_exit", cfg.premature_exit_terminal)
        return breakdown

    if last_customer_features["hostility_score"] > 0:
        breakdown.add("hostile_end", cfg.hostile_end_terminal)
        return breakdown

    if engagement_count >= cfg.min_engagement_signals and slots.filled_count == 0:
        breakdown.add("wasted_engaged", cfg.wasted_engaged_terminal)
        return breakdown

    breakdown.add("neutral_end", 0.0)
    return breakdown


def compute_transition_reward(
    call: Call,
    bot_turn_index: int,
    action: Action,
    *,
    is_last_bot_turn: bool,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """Combine turn reward with terminal reward on the final bot turn."""
    breakdown = compute_turn_reward(call, bot_turn_index, action, config=config)
    if is_last_bot_turn:
        terminal = compute_terminal_reward(call, config=config)
        for name, value in terminal.components.items():
            breakdown.add(name, value)
    return breakdown
