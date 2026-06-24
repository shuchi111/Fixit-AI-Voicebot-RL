#!/usr/bin/env python3
"""Entry point for the voicebot RL pipeline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from collections import Counter

from src.action_infer import action_distribution, infer_action_at_bot_turn
from src.config import load_config
from src.features import FEATURE_NAMES, extract_state_at_bot_turn
from src.models import Speaker
from src.parser import format_parse_report, parse_transcript_dir
from src.policy.baseline import BehaviorPolicy
from src.policy.constraints import PolicyConstraints
from src.policy.discretizer import StateDiscretizer
from src.control.freeze import FreezeRules
from src.control.inspect import diff_policies, format_diff, inspect_policy
from src.control.registry import PolicyRegistry
from src.eval.propensity import PropensityConfig
from src.eval.run_eval import evaluate_policies, load_evaluation_policy
from src.policy.learner import LearnedPolicy, QLearningConfig
from src.reward import RewardConfig
from src.split import split_call_ids, split_trajectories
from src.train import load_policy, save_learning_curve, save_policy, train_policy
from src.trajectory import build_trajectories, summarise_trajectories, trajectories_to_rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def cmd_parse_only(data_dir: Path, config: dict) -> int:
    parser_cfg = config.get("parser", {})
    calls, summary = parse_transcript_dir(
        data_dir,
        timestamp_format=parser_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S UTC"),
        validate_message_count=parser_cfg.get("validate_message_count", True),
    )
    print(format_parse_report(summary))

    outputs_dir = Path(config.get("outputs", {}).get("dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "parse_report.json"
    report_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    print(f"\nReport saved to {report_path}")
    print(f"Sample call SID: {calls[0].call_sid} ({calls[0].message_count} turns)")
    return 0 if not summary.failed_files else 1


def cmd_extract_features(data_dir: Path, config: dict) -> int:
    parser_cfg = config.get("parser", {})
    feature_cfg = config.get("features", {})
    calls, summary = parse_transcript_dir(
        data_dir,
        timestamp_format=parser_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S UTC"),
        validate_message_count=parser_cfg.get("validate_message_count", True),
    )
    if summary.failed_files:
        print(f"Parse failures: {summary.failed_files}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for call in calls:
        for turn in call.turns:
            if turn.speaker is not Speaker.AI_ASSISTANT:
                continue
            state = extract_state_at_bot_turn(call, turn.index)
            row = {
                "call_sid": state.call_sid,
                "turn_index": state.turn_index,
            }
            row.update(state.features)
            rows.append(row)

    frame = pd.DataFrame(rows)
    outputs_dir = Path(config.get("outputs", {}).get("dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    states_path = outputs_dir / feature_cfg.get("states_file", "bot_states.parquet")
    frame.to_parquet(states_path, index=False)

    names_path = outputs_dir / feature_cfg.get("names_file", "feature_names.json")
    names_path.write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    feature_report = {
        "total_calls": len(calls),
        "total_bot_states": len(frame),
        "feature_names": FEATURE_NAMES,
        "feature_means": {name: float(frame[name].mean()) for name in FEATURE_NAMES},
        "nonzero_rates": {
            name: float((frame[name] != 0).mean()) for name in FEATURE_NAMES
        },
    }
    report_path = outputs_dir / feature_cfg.get("report_file", "feature_report.json")
    report_path.write_text(json.dumps(feature_report, indent=2), encoding="utf-8")

    print("Feature extraction complete")
    print(f"  Calls:           {len(calls)}")
    print(f"  Bot state rows:  {len(frame)}")
    print(f"  Features:        {len(FEATURE_NAMES)}")
    print(f"  States saved:    {states_path}")
    print(f"  Report saved:    {report_path}")
    print("\nTop non-zero feature rates:")
    rates = sorted(feature_report["nonzero_rates"].items(), key=lambda x: -x[1])[:8]
    for name, rate in rates:
        print(f"  {name:28s} {rate:6.1%}")
    return 0


def cmd_infer_actions(data_dir: Path, config: dict) -> int:
    parser_cfg = config.get("parser", {})
    action_cfg = config.get("actions", {})
    calls, summary = parse_transcript_dir(
        data_dir,
        timestamp_format=parser_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S UTC"),
        validate_message_count=parser_cfg.get("validate_message_count", True),
    )
    if summary.failed_files:
        print(f"Parse failures: {summary.failed_files}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    all_inferences = []
    for call in calls:
        for turn in call.turns:
            if turn.speaker is not Speaker.AI_ASSISTANT:
                continue
            state = extract_state_at_bot_turn(call, turn.index)
            inference = infer_action_at_bot_turn(call, turn.index)
            all_inferences.append(inference)
            rows.append(
                {
                    "call_sid": call.call_sid,
                    "turn_index": turn.index,
                    "action": inference.action.value,
                    "rule": inference.rule,
                    "bot_text": inference.bot_text,
                    "last_customer_text": inference.last_customer_text or "",
                    "context_mismatch": inference.context_mismatch,
                    **state.features,
                }
            )

    frame = pd.DataFrame(rows)
    outputs_dir = Path(config.get("outputs", {}).get("dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    actions_path = outputs_dir / action_cfg.get("labels_file", "bot_actions.parquet")
    frame.to_parquet(actions_path, index=False)

    mdp_path = outputs_dir / action_cfg.get("mdp_rows_file", "mdp_rows.parquet")
    frame.to_parquet(mdp_path, index=False)

    distribution = action_distribution(all_inferences)
    mismatch_count = sum(1 for inf in all_inferences if inf.context_mismatch)
    action_report = {
        "total_calls": len(calls),
        "total_bot_actions": len(frame),
        "action_distribution": distribution,
        "context_mismatch_count": mismatch_count,
        "context_mismatch_rate": mismatch_count / max(len(all_inferences), 1),
        "top_rules": (
            frame["rule"].value_counts().head(10).astype(int).to_dict()
        ),
    }
    report_path = outputs_dir / action_cfg.get("report_file", "action_report.json")
    report_path.write_text(json.dumps(action_report, indent=2), encoding="utf-8")

    print("Action inference complete")
    print(f"  Calls:              {len(calls)}")
    print(f"  Bot actions:        {len(frame)}")
    print(f"  Context mismatches: {mismatch_count} ({action_report['context_mismatch_rate']:.1%})")
    print(f"  MDP rows saved:     {mdp_path}")
    print(f"  Report saved:       {report_path}")
    print("\nAction distribution:")
    for action, count in sorted(distribution.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {action:22s} {count:6d}")
    return 0


def cmd_build_trajectories(data_dir: Path, config: dict) -> int:
    parser_cfg = config.get("parser", {})
    outputs_cfg = config.get("outputs", {})
    reward_cfg = RewardConfig.from_dict(config.get("reward", {}))

    calls, summary = parse_transcript_dir(
        data_dir,
        timestamp_format=parser_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S UTC"),
        validate_message_count=parser_cfg.get("validate_message_count", True),
    )
    if summary.failed_files:
        print(f"Parse failures: {summary.failed_files}", file=sys.stderr)
        return 1

    trajectories = build_trajectories(calls, reward_config=reward_cfg)
    traj_summary = summarise_trajectories(trajectories)
    rows = trajectories_to_rows(trajectories)

    outputs_dir = Path(outputs_cfg.get("dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)

    trajectories_path = outputs_dir / outputs_cfg.get(
        "trajectories_file", "trajectories.parquet"
    )
    pd.DataFrame(rows).to_parquet(trajectories_path, index=False)

    component_counts: Counter[str] = Counter()
    for row in rows:
        for name in row["reward_breakdown"]:
            component_counts[name] += 1

    reward_report = {
        "baseline_summary": traj_summary.to_dict(),
        "reward_component_counts": dict(component_counts),
        "mean_reward_per_transition": float(
            pd.DataFrame(rows)["reward"].mean()
        ),
    }
    reward_report_path = outputs_dir / "reward_report.json"
    reward_report_path.write_text(json.dumps(reward_report, indent=2), encoding="utf-8")

    print("Trajectory building complete")
    print(f"  Calls:                   {traj_summary.total_calls}")
    print(f"  Transitions:             {traj_summary.total_transitions}")
    print(f"  Baseline mean return:    {traj_summary.mean_return:.3f}")
    print(f"  Return range:            [{traj_summary.min_return:.3f}, {traj_summary.max_return:.3f}]")
    print(f"  Mean reward/transition:  {reward_report['mean_reward_per_transition']:.3f}")
    print(f"  Trajectories saved:      {trajectories_path}")
    print(f"  Reward report saved:     {reward_report_path}")
    print("\nTop reward components:")
    for name, count in component_counts.most_common(8):
        print(f"  {name:22s} {count:6d}")
    return 0


def cmd_train(data_dir: Path, config: dict) -> int:
    parser_cfg = config.get("parser", {})
    outputs_cfg = config.get("outputs", {})
    split_cfg = config.get("split", {})
    policy_cfg = config.get("policy", {})
    control_cfg = config.get("control", {})
    reward_cfg = RewardConfig.from_dict(config.get("reward", {}))

    calls, summary = parse_transcript_dir(
        data_dir,
        timestamp_format=parser_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S UTC"),
        validate_message_count=parser_cfg.get("validate_message_count", True),
    )
    if summary.failed_files:
        print(f"Parse failures: {summary.failed_files}", file=sys.stderr)
        return 1

    trajectories = build_trajectories(calls, reward_config=reward_cfg)
    data_split = split_call_ids(
        [call.call_sid for call in calls],
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.10)),
        test_ratio=float(split_cfg.get("test_ratio", 0.20)),
        seed=int(config.get("seed", 42)),
    )

    discretizer = StateDiscretizer.from_dict(config.get("discretizer", {}))
    q_config = QLearningConfig.from_dict(config.get("learning", {}))
    constraints_data = config.get("constraints", {})
    constraints = PolicyConstraints(
        max_kl_divergence=float(constraints_data.get("max_kl_divergence", 0.8)),
        blend_alpha=float(constraints_data.get("blend_alpha", 0.5)),
        min_q_improvement=float(constraints_data.get("min_q_improvement", 0.12)),
    )

    result = train_policy(
        trajectories,
        data_split,
        discretizer=discretizer,
        q_config=q_config,
        constraints=constraints,
    )

    outputs_dir = Path(outputs_cfg.get("dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    policies_dir = Path("policies")
    policies_dir.mkdir(parents=True, exist_ok=True)

    learned_path = policies_dir / policy_cfg.get(
        "learned_file", "policy_learned_v1.json"
    )
    deploy_file = control_cfg.get(
        "deploy_file", policy_cfg.get("learned_file", "policy_learned_v1.json")
    )
    freeze_path = Path(config.get("control", {}).get(
        "freeze_rules_file", "configs/freeze_rules.yaml"
    ))
    if freeze_path.exists():
        result.policy.freeze_rules = FreezeRules.from_yaml(freeze_path)

    result.policy.version = learned_path.stem.replace("policy_", "")
    registry = PolicyRegistry.load(policies_dir)
    train_traj, _, _ = split_trajectories(trajectories, data_split)
    behavior_policy = BehaviorPolicy().fit(train_traj, discretizer)
    baseline_path = policies_dir / policy_cfg.get(
        "baseline_file", "policy_baseline_v0.json"
    )
    baseline_policy = LearnedPolicy(
        q_values={},
        discretizer=discretizer,
        behavior_policy=behavior_policy,
        version="baseline_v0",
        metadata={"baseline_val_return": result.baseline_val_return},
    )
    save_policy(baseline_policy, baseline_path)
    registry.register(baseline_policy, baseline_path.name, parent_version=None)

    save_policy(result.policy, learned_path)
    registry.register(
        result.policy,
        learned_path.name,
        parent_version="baseline_v0",
    )
    registry.rollback(result.policy.version, deploy_file=deploy_file)

    curve_path = outputs_dir / outputs_cfg.get(
        "learning_curve_file", "learning_curve.png"
    )
    save_learning_curve(result.learning_curve, curve_path)

    train_report = result.to_dict()
    train_report_path = outputs_dir / "train_report.json"
    train_report_path.write_text(json.dumps(train_report, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"  Train calls:          {len(data_split.train)}")
    print(f"  Val calls:            {len(data_split.val)}")
    print(f"  Baseline val return:  {result.baseline_val_return:.3f}")
    print(f"  Learned val return:   {result.learned_val_return:.3f}")
    print(f"  Improvement:          {result.learned_val_return - result.baseline_val_return:.3f}")
    print(f"  Policy saved:         {learned_path}")
    print(f"  Active deploy file:   {policies_dir / deploy_file}")
    print(f"  Learning curve:       {curve_path}")
    print(f"  Train report:         {train_report_path}")

    if result.learned_val_return < result.baseline_val_return:
        print(
            "\nWarning: learned policy did not beat baseline on validation replay.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_eval(data_dir: Path, config: dict, policy_path: Path | None = None) -> int:
    parser_cfg = config.get("parser", {})
    outputs_cfg = config.get("outputs", {})
    split_cfg = config.get("split", {})
    policy_cfg = config.get("policy", {})
    ope_cfg = config.get("ope", {})
    reward_cfg = RewardConfig.from_dict(config.get("reward", {}))

    calls, summary = parse_transcript_dir(
        data_dir,
        timestamp_format=parser_cfg.get("timestamp_format", "%Y-%m-%d %H:%M:%S UTC"),
        validate_message_count=parser_cfg.get("validate_message_count", True),
    )
    if summary.failed_files:
        print(f"Parse failures: {summary.failed_files}", file=sys.stderr)
        return 1

    trajectories = build_trajectories(calls, reward_config=reward_cfg)
    data_split = split_call_ids(
        [call.call_sid for call in calls],
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.10)),
        test_ratio=float(split_cfg.get("test_ratio", 0.20)),
        seed=int(config.get("seed", 42)),
    )

    discretizer = StateDiscretizer.from_dict(config.get("discretizer", {}))
    policies_dir = Path("policies")
    if policy_path is not None:
        resolved_policy = policy_path
    else:
        registry = PolicyRegistry.load(policies_dir)
        if registry.active_file:
            resolved_policy = registry.active_policy_path()
        else:
            resolved_policy = policies_dir / policy_cfg.get(
                "learned_file", "policy_learned_v1.json"
            )
    evaluation_policy = load_evaluation_policy(resolved_policy)

    propensity_config = PropensityConfig(
        min_propensity=float(ope_cfg.get("min_propensity", 0.01)),
        rho_max=float(ope_cfg.get("rho_max", 10.0)),
    )
    eval_report = evaluate_policies(
        trajectories,
        data_split,
        evaluation_policy,
        discretizer,
        ope_config=propensity_config,
        bootstrap_samples=int(ope_cfg.get("bootstrap_samples", 1000)),
        seed=int(config.get("seed", 42)),
    )

    outputs_dir = Path(outputs_cfg.get("dir", "outputs"))
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / outputs_cfg.get("eval_report_file", "eval_report.json")
    report_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    dr = eval_report["dr"]
    ips = eval_report["ips"]
    print("Offline policy evaluation complete")
    print(f"  Test episodes:        {eval_report['n_test_episodes']}")
    print(f"  Baseline test return: {eval_report['baseline_test_return']:.3f}")
    print(f"  IPS estimate:         {ips['point_estimate']:.3f}  CI {ips['ci_95']}")
    print(f"  DR estimate:          {dr['point_estimate']:.3f}  CI {dr['ci_95']}")
    print(f"  Improvement (DR):     {eval_report['improvement_vs_baseline']:.3f}")
    print(f"  Significant:          {eval_report['significant_improvement']}")
    print(f"  Report saved:         {report_path}")
    print("\nAssumptions:")
    for item in dr["assumptions"]:
        print(f"  - {item}")
    return 0


def cmd_rollback(config: dict, version: str) -> int:
    control_cfg = config.get("control", {})
    policy_cfg = config.get("policy", {})
    policies_dir = Path("policies")
    deploy_file = control_cfg.get(
        "deploy_file", policy_cfg.get("learned_file", "policy_learned_v1.json")
    )

    registry = PolicyRegistry.load(policies_dir)
    try:
        deploy_path = registry.rollback(version, deploy_file=deploy_file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Rollback failed: {exc}", file=sys.stderr)
        return 1

    print("Rollback complete")
    print(f"  Active version:  {registry.active_version}")
    print(f"  Deployed file:   {deploy_path}")
    return 0


def cmd_freeze(config: dict, freeze_path: Path) -> int:
    control_cfg = config.get("control", {})
    policy_cfg = config.get("policy", {})
    policies_dir = Path("policies")
    deploy_file = control_cfg.get(
        "deploy_file", policy_cfg.get("learned_file", "policy_learned_v1.json")
    )

    if not freeze_path.exists():
        print(f"Freeze rules not found: {freeze_path}", file=sys.stderr)
        return 1

    deploy_path = policies_dir / deploy_file
    if not deploy_path.exists():
        print(f"Active policy not found: {deploy_path}. Run --train first.", file=sys.stderr)
        return 1

    policy = load_policy(deploy_path)
    policy.freeze_rules = FreezeRules.from_yaml(freeze_path)
    save_policy(policy, deploy_path)

    registry = PolicyRegistry.load(policies_dir)
    registry.active_file = deploy_file
    registry.save()
    print("Freeze rules applied")
    print(f"  Policy:          {deploy_path}")
    print(f"  Rules file:      {freeze_path}")
    print(f"  Force rules:     {len(policy.freeze_rules.force_action_rules)}")
    return 0


def cmd_inspect(
    config: dict,
    *,
    policy_path: Path | None = None,
    diff_left: str | None = None,
    diff_right: str | None = None,
) -> int:
    policies_dir = Path("policies")
    policy_cfg = config.get("policy", {})

    if diff_left and diff_right:
        registry = PolicyRegistry.load(policies_dir)
        left_record = registry.find_record(diff_left)
        right_record = registry.find_record(diff_right)
        if left_record is None or right_record is None:
            print("Could not resolve one or both policy versions", file=sys.stderr)
            return 1
        left = load_policy(policies_dir / left_record.file_name)
        right = load_policy(policies_dir / right_record.file_name)
        changes = diff_policies(left, right)
        print(format_diff(changes))
        return 0

    if policy_path is not None:
        resolved = policy_path
    else:
        registry = PolicyRegistry.load(policies_dir)
        if registry.active_file:
            resolved = registry.active_policy_path()
        else:
            resolved = policies_dir / policy_cfg.get(
                "learned_file", "policy_learned_v1.json"
            )

    if not resolved.exists():
        print(f"Policy not found: {resolved}", file=sys.stderr)
        return 1

    policy = load_policy(resolved)
    print(inspect_policy(policy).format_summary())
    return 0


def cmd_report(config: dict) -> int:
    outputs_dir = Path(config.get("outputs", {}).get("dir", "outputs"))
    train_report_path = outputs_dir / "train_report.json"
    eval_report_path = outputs_dir / config.get("outputs", {}).get(
        "eval_report_file", "eval_report.json"
    )

    missing = [path for path in (train_report_path, eval_report_path) if not path.exists()]
    if missing:
        print(
            "Report requires train and eval outputs. Missing:",
            ", ".join(str(path) for path in missing),
            file=sys.stderr,
        )
        print("Run: uv run python run.py --train --eval --report", file=sys.stderr)
        return 1

    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    eval_report = json.loads(eval_report_path.read_text(encoding="utf-8"))

    policies_dir = Path("policies")
    registry = PolicyRegistry.load(policies_dir)
    final_report = {
        "active_policy_version": registry.active_version,
        "active_policy_file": registry.active_file,
        "training": {
            "baseline_val_return": train_report["baseline_val_return"],
            "learned_val_return": train_report["learned_val_return"],
            "improvement": train_report["improvement"],
            "data_split": train_report["data_split"],
        },
        "offline_evaluation": {
            "primary_estimator": eval_report["primary_estimator"],
            "baseline_test_return": eval_report["baseline_test_return"],
            "dr_estimate": eval_report["dr"]["point_estimate"],
            "dr_ci_95": eval_report["dr"]["ci_95"],
            "improvement_vs_baseline": eval_report["improvement_vs_baseline"],
            "significant_improvement": eval_report["significant_improvement"],
            "n_test_episodes": eval_report["n_test_episodes"],
        },
        "comparison_table": [
            {
                "metric": "Validation replay return",
                "baseline": train_report["baseline_val_return"],
                "learned": train_report["learned_val_return"],
            },
            {
                "metric": "Test DR estimate",
                "baseline": eval_report["baseline_test_return"],
                "learned": eval_report["dr"]["point_estimate"],
            },
        ],
    }

    report_path = outputs_dir / "final_report.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print("Final report generated")
    print(f"  Active policy:        {registry.active_version or 'n/a'}")
    print(f"  Val baseline/learned: {train_report['baseline_val_return']:.3f} / "
          f"{train_report['learned_val_return']:.3f}")
    print(f"  Test DR estimate:     {eval_report['dr']['point_estimate']:.3f} "
          f"CI {eval_report['dr']['ci_95']}")
    print(f"  Report saved:         {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Voicebot RL — learn dialogue policy from call transcripts",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Directory containing transcript .txt files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse transcripts and print statistics",
    )
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Parse transcripts and extract bot decision state features",
    )
    parser.add_argument(
        "--infer-actions",
        action="store_true",
        help="Parse transcripts, extract states, and infer bot actions",
    )
    parser.add_argument(
        "--build-trajectories",
        action="store_true",
        help="Build full MDP trajectories with rewards (baseline policy)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train tabular Q-learning policy and save learning curve",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Offline policy evaluation (IPS + DR with bootstrap CIs)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to learned policy JSON for --eval",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate final report from train + eval outputs",
    )
    parser.add_argument(
        "--rollback",
        type=str,
        default=None,
        metavar="VERSION",
        help="Rollback active deploy policy to a registered version",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        nargs="?",
        const=Path("configs/freeze_rules.yaml"),
        help="Apply freeze rules YAML to the active deploy policy",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect active or --policy JSON and print action summary",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("LEFT", "RIGHT"),
        help="Diff two registered policy versions (use with --inspect)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)

    data_dir = args.data or Path(config["data"]["transcripts_dir"])
    if not data_dir.is_dir():
        print(f"Error: transcript directory not found: {data_dir}", file=sys.stderr)
        return 1

    pipeline_flags = (
        args.train,
        args.eval,
        args.report,
        args.rollback,
        args.freeze is not None,
        args.inspect,
    )
    step_flags = (
        args.parse_only,
        args.extract_features,
        args.infer_actions,
        args.build_trajectories,
    )

    if not any(pipeline_flags) and not any(step_flags):
        return cmd_parse_only(data_dir, config)

    exit_code = 0

    if args.parse_only:
        return cmd_parse_only(data_dir, config)
    if args.extract_features:
        return cmd_extract_features(data_dir, config)
    if args.infer_actions:
        return cmd_infer_actions(data_dir, config)
    if args.build_trajectories:
        return cmd_build_trajectories(data_dir, config)

    if args.train:
        exit_code = max(exit_code, cmd_train(data_dir, config))
    if args.freeze is not None:
        exit_code = max(exit_code, cmd_freeze(config, args.freeze))
    if args.rollback:
        exit_code = max(exit_code, cmd_rollback(config, args.rollback))
    if args.inspect:
        diff_left, diff_right = (args.diff if args.diff else (None, None))
        exit_code = max(
            exit_code,
            cmd_inspect(
                config,
                policy_path=args.policy,
                diff_left=diff_left,
                diff_right=diff_right,
            ),
        )
    if args.eval:
        exit_code = max(exit_code, cmd_eval(data_dir, config, policy_path=args.policy))
    if args.report:
        exit_code = max(exit_code, cmd_report(config))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
