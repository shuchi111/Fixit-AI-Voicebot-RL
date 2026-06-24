"""Build MDP trajectories from parsed calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.action_infer import infer_action_at_bot_turn
from src.features import extract_state_at_bot_turn
from src.models import Call, Speaker, Transition, Trajectory
from src.reward import RewardConfig, compute_transition_reward


@dataclass
class TrajectorySummary:
    """Aggregate statistics over built trajectories."""

    total_calls: int
    total_transitions: int
    mean_return: float
    min_return: float
    max_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_transitions": self.total_transitions,
            "mean_return": self.mean_return,
            "min_return": self.min_return,
            "max_return": self.max_return,
        }


def build_trajectory_for_call(
    call: Call,
    *,
    reward_config: RewardConfig | None = None,
) -> Trajectory:
    """Build (s, a, r, s', done) transitions for one call episode."""
    bot_turn_indices = [
        turn.index for turn in call.turns if turn.speaker is Speaker.AI_ASSISTANT
    ]
    transitions: list[Transition] = []

    for ordinal, bot_index in enumerate(bot_turn_indices):
        state = extract_state_at_bot_turn(call, bot_index)
        inference = infer_action_at_bot_turn(call, bot_index)
        is_last = ordinal == len(bot_turn_indices) - 1
        reward_breakdown = compute_transition_reward(
            call,
            bot_index,
            inference.action,
            is_last_bot_turn=is_last,
            config=reward_config,
        )

        next_state = None
        if not is_last:
            next_bot_index = bot_turn_indices[ordinal + 1]
            next_state = extract_state_at_bot_turn(call, next_bot_index)

        transitions.append(
            Transition(
                call_sid=call.call_sid,
                turn_index=bot_index,
                state=state,
                action=inference.action,
                reward=reward_breakdown.total,
                next_state=next_state,
                done=is_last,
                reward_breakdown=dict(reward_breakdown.components),
            )
        )

    return Trajectory(call_sid=call.call_sid, transitions=transitions)


def build_trajectories(
    calls: list[Call],
    *,
    reward_config: RewardConfig | None = None,
) -> list[Trajectory]:
    return [build_trajectory_for_call(call, reward_config=reward_config) for call in calls]


def summarise_trajectories(trajectories: list[Trajectory]) -> TrajectorySummary:
    returns = [traj.total_return for traj in trajectories]
    if not returns:
        return TrajectorySummary(0, 0, 0.0, 0.0, 0.0)
    return TrajectorySummary(
        total_calls=len(trajectories),
        total_transitions=sum(len(t.transitions) for t in trajectories),
        mean_return=sum(returns) / len(returns),
        min_return=min(returns),
        max_return=max(returns),
    )


def trajectories_to_rows(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    """Flatten trajectories into tabular rows for parquet export."""
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            row: dict[str, Any] = {
                "call_sid": transition.call_sid,
                "turn_index": transition.turn_index,
                "action": transition.action.value if transition.action else "",
                "reward": transition.reward,
                "done": transition.done,
                "reward_breakdown": transition.reward_breakdown,
            }
            row.update(transition.state.features)
            rows.append(row)
    return rows
