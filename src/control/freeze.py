"""Apply human-defined freeze and force-action rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.models import Action


@dataclass
class ForceActionRule:
    when_state_contains: str
    action: Action
    reason: str = ""


@dataclass
class FreezeRules:
    force_action_rules: list[ForceActionRule] = field(default_factory=list)
    never_override_actions: list[Action] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreezeRules:
        rules = [
            ForceActionRule(
                when_state_contains=str(item["when_state_contains"]),
                action=Action(item["action"]),
                reason=str(item.get("reason", "")),
            )
            for item in data.get("force_action_rules", [])
        ]
        never_override = [
            Action(name) for name in data.get("never_override_actions", [])
        ]
        return cls(force_action_rules=rules, never_override_actions=never_override)

    @classmethod
    def from_yaml(cls, path: Path | str) -> FreezeRules:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_action_rules": [
                {
                    "when_state_contains": rule.when_state_contains,
                    "action": rule.action.value,
                    "reason": rule.reason,
                }
                for rule in self.force_action_rules
            ],
            "never_override_actions": [
                action.value for action in self.never_override_actions
            ],
        }


def apply_freeze_rules(
    state_key: str,
    proposed_action: Action,
    rules: FreezeRules | None,
) -> tuple[Action, str | None]:
    """Return action after applying force/never-override rules."""
    if rules is None:
        return proposed_action, None

    for rule in rules.force_action_rules:
        if rule.when_state_contains in state_key:
            return rule.action, rule.reason or rule.when_state_contains

    if proposed_action in rules.never_override_actions:
        return proposed_action, "never_override"

    return proposed_action, None
