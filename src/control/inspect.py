"""Inspect policies and compare versions for human review."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.models import Action
from src.policy.learner import LearnedPolicy
from src.train import load_policy


@dataclass
class PolicyInspection:
    version: str
    num_states: int
    action_distribution: dict[str, int]
    top_states: list[dict[str, Any]]

    def format_summary(self) -> str:
        lines = [
            f"Policy version: {self.version}",
            f"State clusters: {self.num_states}",
            "Action distribution:",
        ]
        for action, count in sorted(
            self.action_distribution.items(), key=lambda item: -item[1]
        ):
            lines.append(f"  {action:22s} {count}")
        lines.append("Top states:")
        for item in self.top_states[:10]:
            lines.append(
                f"  {item['state_key']:30s} -> {item['action']} (Q={item['max_q']:.3f})"
            )
        return "\n".join(lines)


def _greedy_actions(policy: LearnedPolicy) -> dict[str, Action]:
    return {
        state_key: policy.select_action(state_key) for state_key in policy.q_values
    }


def inspect_policy(policy: LearnedPolicy) -> PolicyInspection:
    actions = _greedy_actions(policy)
    distribution = Counter(action.value for action in actions.values())
    top_states = []
    for state_key, state_q in policy.q_values.items():
        max_q = max(state_q.values()) if state_q else 0.0
        top_states.append(
            {
                "state_key": state_key,
                "action": actions[state_key].value,
                "max_q": max_q,
            }
        )
    top_states.sort(key=lambda item: item["max_q"], reverse=True)
    return PolicyInspection(
        version=policy.version,
        num_states=len(policy.q_values),
        action_distribution=dict(distribution),
        top_states=top_states,
    )


def diff_policies(left: LearnedPolicy, right: LearnedPolicy) -> list[dict[str, Any]]:
    left_actions = _greedy_actions(left)
    right_actions = _greedy_actions(right)
    changes: list[dict[str, Any]] = []
    for state_key in set(left_actions) | set(right_actions):
        old = left_actions.get(state_key)
        new = right_actions.get(state_key)
        if old is None or new is None or old is not new:
            changes.append(
                {
                    "state_key": state_key,
                    "old_action": old.value if old else None,
                    "new_action": new.value if new else None,
                    "delta_q": right.q(state_key, new or Action.CLARIFY_REPEAT)
                    - left.q(state_key, old or Action.CLARIFY_REPEAT),
                }
            )
    changes.sort(key=lambda item: abs(item["delta_q"]), reverse=True)
    return changes


def format_diff(changes: list[dict[str, Any]], limit: int = 10) -> str:
    lines = [f"Changed states: {len(changes)}", "Top changes:"]
    for item in changes[:limit]:
        lines.append(
            f"  {item['state_key']:30s} "
            f"{item['old_action']} -> {item['new_action']} "
            f"(delta_q={item['delta_q']:.3f})"
        )
    return "\n".join(lines)


def load_policy_file(path: Path) -> LearnedPolicy:
    return load_policy(path)
