"""Versioned policy registry with rollback support."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.policy.learner import LearnedPolicy
from src.train import load_policy, save_policy


@dataclass
class PolicyRecord:
    version: str
    file_name: str
    parent_version: str | None = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "file_name": self.file_name,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyRecord:
        return cls(
            version=str(data["version"]),
            file_name=str(data["file_name"]),
            parent_version=data.get("parent_version"),
            created_at=str(data.get("created_at", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class PolicyRegistry:
    policies_dir: Path
    active_file: str
    active_version: str
    records: list[PolicyRecord] = field(default_factory=list)

    @property
    def registry_path(self) -> Path:
        return self.policies_dir / "registry.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_file": self.active_file,
            "active_version": self.active_version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], policies_dir: Path) -> PolicyRegistry:
        return cls(
            policies_dir=policies_dir,
            active_file=str(data.get("active_file", "")),
            active_version=str(data.get("active_version", "")),
            records=[
                PolicyRecord.from_dict(item) for item in data.get("records", [])
            ],
        )

    @classmethod
    def load(cls, policies_dir: Path) -> PolicyRegistry:
        path = policies_dir / "registry.json"
        if not path.exists():
            return cls(
                policies_dir=policies_dir,
                active_file="",
                active_version="",
                records=[],
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, policies_dir)

    def save(self) -> None:
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    def register(
        self,
        policy: LearnedPolicy,
        file_name: str,
        *,
        parent_version: str | None = None,
    ) -> PolicyRecord:
        record = PolicyRecord(
            version=policy.version,
            file_name=file_name,
            parent_version=parent_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=dict(policy.metadata),
        )
        self.records = [r for r in self.records if r.version != policy.version]
        self.records.append(record)
        self.active_file = file_name
        self.active_version = policy.version
        self.save()
        return record

    def find_record(self, version: str) -> PolicyRecord | None:
        aliases = {
            "baseline": "baseline_v0",
            "baseline_v0": "baseline_v0",
            "learned_v1": "learned_v1",
            "v1": "learned_v1",
        }
        target = aliases.get(version, version)
        for record in self.records:
            if record.version == target or record.file_name == version:
                return record
        return None

    def active_policy_path(self) -> Path:
        if self.active_file:
            return self.policies_dir / self.active_file
        raise FileNotFoundError("No active policy registered")

    def rollback(self, version: str, *, deploy_file: str) -> Path:
        record = self.find_record(version)
        if record is None:
            raise ValueError(f"Unknown policy version: {version}")

        source = self.policies_dir / record.file_name
        if not source.exists():
            raise FileNotFoundError(f"Policy file missing: {source}")

        deploy_path = self.policies_dir / deploy_file
        shutil.copy2(source, deploy_path)
        self.active_file = deploy_file
        self.active_version = record.version
        self.save()
        return deploy_path

    def load_active_policy(self) -> LearnedPolicy:
        return load_policy(self.active_policy_path())
