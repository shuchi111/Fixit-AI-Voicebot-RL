"""Tests for human control: freeze rules, registry, inspect."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.control.freeze import FreezeRules, apply_freeze_rules
from src.control.inspect import diff_policies, inspect_policy
from src.control.registry import PolicyRegistry
from src.models import Action
from src.policy.baseline import BehaviorPolicy
from src.policy.constraints import PolicyConstraints
from src.policy.discretizer import StateDiscretizer
from src.policy.learner import LearnedPolicy
from src.train import load_policy, save_policy


HOSTILE_STATE = "s1|o0|b0|e0|h1|c0|r0|t2"
BUSY_STATE = "s0|o0|b1|e0|h0|c0|r0|t1"
NEUTRAL_STATE = "s1|o0|b0|e1|h0|c0|r0|t2"


class TestFreezeRules:
    def test_force_graceful_exit_on_hostility(self) -> None:
        rules = FreezeRules.from_yaml(Path("configs/freeze_rules.yaml"))
        action, reason = apply_freeze_rules(
            HOSTILE_STATE,
            Action.ASK_BUDGET,
            rules,
        )
        assert action is Action.GRACEFUL_EXIT
        assert reason

    def test_force_busy_defer(self) -> None:
        rules = FreezeRules.from_yaml(Path("configs/freeze_rules.yaml"))
        action, _ = apply_freeze_rules(BUSY_STATE, Action.ASK_LOCATION, rules)
        assert action is Action.ACK_BUSY_DEFER

    def test_no_rule_leaves_action_unchanged(self) -> None:
        rules = FreezeRules.from_yaml(Path("configs/freeze_rules.yaml"))
        action, reason = apply_freeze_rules(
            NEUTRAL_STATE,
            Action.ASK_BUDGET,
            rules,
        )
        assert action is Action.ASK_BUDGET
        assert reason is None


class TestLearnedPolicyFreeze:
    def test_select_action_applies_freeze_rules(self) -> None:
        rules = FreezeRules.from_yaml(Path("configs/freeze_rules.yaml"))
        policy = LearnedPolicy(
            q_values={
                HOSTILE_STATE: {Action.ASK_BUDGET.value: 1.0},
                BUSY_STATE: {Action.ASK_LOCATION.value: 1.0},
            },
            freeze_rules=rules,
        )
        assert policy.select_action(HOSTILE_STATE) is Action.GRACEFUL_EXIT
        assert policy.select_action(BUSY_STATE) is Action.ACK_BUSY_DEFER


class TestPolicyRegistry:
    def test_register_and_rollback(self, tmp_path: Path) -> None:
        discretizer = StateDiscretizer()
        baseline = LearnedPolicy(
            q_values={},
            discretizer=discretizer,
            behavior_policy=BehaviorPolicy(),
            version="baseline_v0",
        )
        learned = LearnedPolicy(
            q_values={NEUTRAL_STATE: {Action.ASK_BUDGET.value: 0.5}},
            discretizer=discretizer,
            version="learned_v1",
        )

        baseline_path = tmp_path / "policy_baseline_v0.json"
        learned_path = tmp_path / "policy_learned_v1.json"
        save_policy(baseline, baseline_path)
        save_policy(learned, learned_path)

        registry = PolicyRegistry.load(tmp_path)
        registry.register(baseline, baseline_path.name)
        registry.register(learned, learned_path.name, parent_version="baseline_v0")

        deploy_path = registry.rollback("baseline", deploy_file=learned_path.name)
        assert deploy_path == learned_path
        assert registry.active_version == "baseline_v0"
        assert load_policy(deploy_path).version == "baseline_v0"

        registry_path = tmp_path / "registry.json"
        saved = json.loads(registry_path.read_text(encoding="utf-8"))
        assert saved["active_version"] == "baseline_v0"


class TestInspect:
    def test_inspect_and_diff(self) -> None:
        discretizer = StateDiscretizer()
        behavior = BehaviorPolicy()
        behavior.update(NEUTRAL_STATE, Action.ASK_BUDGET)
        constraints = PolicyConstraints(min_q_improvement=0.1, max_kl_divergence=5.0)
        left = LearnedPolicy(
            q_values={NEUTRAL_STATE: {Action.ASK_BUDGET.value: 1.0}},
            discretizer=discretizer,
            behavior_policy=behavior,
            constraints=constraints,
            version="baseline_v0",
        )
        right = LearnedPolicy(
            q_values={
                NEUTRAL_STATE: {
                    Action.ASK_BUDGET.value: 1.0,
                    Action.HANDLE_OBJECTION.value: 2.0,
                }
            },
            discretizer=discretizer,
            behavior_policy=behavior,
            constraints=constraints,
            version="learned_v1",
        )

        inspection = inspect_policy(right)
        assert inspection.version == "learned_v1"
        assert inspection.num_states == 1

        changes = diff_policies(left, right)
        assert changes
        assert changes[0]["state_key"] == NEUTRAL_STATE
