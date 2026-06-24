"""CLI: python -m src.control inspect --policy ..."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.control.inspect import diff_policies, inspect_policy, load_policy_file
from src.control.registry import PolicyRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and diff voicebot policies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one policy")
    inspect_parser.add_argument("--policy", type=Path, required=True)

    diff_parser = subparsers.add_parser("diff", help="Diff two policy versions")
    diff_parser.add_argument("left", type=str)
    diff_parser.add_argument("right", type=str)
    diff_parser.add_argument("--policies-dir", type=Path, default=Path("policies"))

    list_parser = subparsers.add_parser("list", help="List registered versions")
    list_parser.add_argument("--policies-dir", type=Path, default=Path("policies"))

    args = parser.parse_args(argv)

    if args.command == "inspect":
        policy = load_policy_file(args.policy)
        print(inspect_policy(policy).format_summary())
        return 0

    if args.command == "list":
        registry = PolicyRegistry.load(args.policies_dir)
        print(json.dumps(registry.to_dict(), indent=2))
        return 0

    registry = PolicyRegistry.load(args.policies_dir)
    left_record = registry.find_record(args.left)
    right_record = registry.find_record(args.right)
    if left_record is None or right_record is None:
        raise SystemExit("Could not resolve one or both policy versions")
    left = load_policy_file(args.policies_dir / left_record.file_name)
    right = load_policy_file(args.policies_dir / right_record.file_name)
    changes = diff_policies(left, right)
    print(__import__("src.control.inspect", fromlist=["format_diff"]).format_diff(changes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
