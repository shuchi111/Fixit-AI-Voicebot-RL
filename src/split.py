"""Split calls into train/val/test without turn-level leakage."""

from __future__ import annotations

from dataclasses import dataclass

from src.models import Call, Trajectory


@dataclass(frozen=True)
class DataSplit:
    train: list[str]
    val: list[str]
    test: list[str]


def split_call_ids(
    call_ids: list[str],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> DataSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    ids = sorted(call_ids)
    rng = __import__("random").Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return DataSplit(
        train=ids[:train_end],
        val=ids[train_end:val_end],
        test=ids[val_end:],
    )


def split_calls(calls: list[Call], data_split: DataSplit) -> tuple[list[Call], list[Call], list[Call]]:
    train_set = set(data_split.train)
    val_set = set(data_split.val)
    test_set = set(data_split.test)
    train_calls = [c for c in calls if c.call_sid in train_set]
    val_calls = [c for c in calls if c.call_sid in val_set]
    test_calls = [c for c in calls if c.call_sid in test_set]
    return train_calls, val_calls, test_calls


def split_trajectories(
    trajectories: list[Trajectory],
    data_split: DataSplit,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    train_set = set(data_split.train)
    val_set = set(data_split.val)
    test_set = set(data_split.test)
    train_traj = [t for t in trajectories if t.call_sid in train_set]
    val_traj = [t for t in trajectories if t.call_sid in val_set]
    test_traj = [t for t in trajectories if t.call_sid in test_set]
    return train_traj, val_traj, test_traj
