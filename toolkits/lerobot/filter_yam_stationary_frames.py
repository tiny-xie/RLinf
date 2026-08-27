#!/usr/bin/env python3
"""Filter truly stationary frames from a dual-YAM LeRobot v2.1 dataset.

A frame is stationary only when both the 14-D commanded action and the 14-D
measured follower state are unchanged within their respective thresholds. Long
stationary runs are downsampled instead of removed completely. The source is
never modified; the filtered dataset is written through a staging directory.

Example:
    python3 filter_yam_stationary_frames.py /path/to/pick_block \
        -o /path/to/pick_block_filtered --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import filter_zero_action_frames as common
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

YAM_VECTOR_DIM = 14
YAM_GRIPPER_INDICES = frozenset((6, 13))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--action-column", default="actions")
    parser.add_argument("--state-column", default="state")
    parser.add_argument("--done-column", default="done")
    parser.add_argument(
        "--action-joint-epsilon",
        type=float,
        default=1e-5,
        help="Maximum per-frame absolute action-joint change (default: 1e-5).",
    )
    parser.add_argument(
        "--action-gripper-epsilon",
        type=float,
        default=1e-5,
        help="Maximum per-frame absolute action-gripper change (default: 1e-5).",
    )
    parser.add_argument(
        "--state-joint-epsilon",
        type=float,
        default=1e-4,
        help="Maximum measured follower joint change (default: 1e-4).",
    )
    parser.add_argument(
        "--state-gripper-epsilon",
        type=float,
        default=1e-4,
        help="Maximum measured follower gripper change (default: 1e-4).",
    )
    parser.add_argument(
        "--stationary-keep-every",
        type=int,
        default=6,
        help="Keep every Nth frame inside a stationary run (default: 6).",
    )
    parser.add_argument(
        "--image-stat-samples",
        type=int,
        default=1,
        help="Image samples per episode/view when rebuilding metadata (default: 1).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_vector(values: Sequence[object], *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (YAM_VECTOR_DIM,):
        raise ValueError(
            f"{name} must use the dual-YAM 14-D layout, got {vector.shape}."
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return vector


def _vector_is_stationary(
    current: Sequence[object],
    previous: Sequence[object],
    *,
    joint_epsilon: float,
    gripper_epsilon: float,
    name: str,
) -> bool:
    delta = np.abs(
        _validate_vector(current, name=name) - _validate_vector(previous, name=name)
    )
    thresholds = np.full(YAM_VECTOR_DIM, joint_epsilon, dtype=np.float64)
    thresholds[list(YAM_GRIPPER_INDICES)] = gripper_epsilon
    return bool(np.all(delta <= thresholds))


def _done(value: object) -> bool:
    """Handle scalar and LeRobot ``shape=(1,)`` done columns correctly."""
    if value is None:
        return False
    values = np.asarray(value, dtype=bool).reshape(-1)
    return bool(values.any())


def build_keep_mask(
    actions: Sequence[Sequence[object] | None],
    states: Sequence[Sequence[object] | None],
    done_values: Sequence[object],
    *,
    action_joint_epsilon: float,
    action_gripper_epsilon: float,
    state_joint_epsilon: float,
    state_gripper_epsilon: float,
    stationary_keep_every: int,
) -> list[bool]:
    """Keep motion frames and periodically retain frames in true holds."""
    if not (len(actions) == len(states) == len(done_values)):
        raise ValueError("actions, states and done columns have different lengths")

    keep_mask: list[bool] = []
    previous_action: Sequence[object] | None = None
    previous_state: Sequence[object] | None = None
    stationary_run = 0
    last_index = len(actions) - 1

    for index, (action, state, done) in enumerate(
        zip(actions, states, done_values, strict=True)
    ):
        stationary = False
        if (
            action is not None
            and state is not None
            and previous_action is not None
            and previous_state is not None
        ):
            action_stationary = _vector_is_stationary(
                action,
                previous_action,
                joint_epsilon=action_joint_epsilon,
                gripper_epsilon=action_gripper_epsilon,
                name="actions",
            )
            state_stationary = _vector_is_stationary(
                state,
                previous_state,
                joint_epsilon=state_joint_epsilon,
                gripper_epsilon=state_gripper_epsilon,
                name="state",
            )
            stationary = action_stationary and state_stationary

        if stationary:
            stationary_run += 1
        else:
            stationary_run = 0

        structural = index == 0 or index == last_index or _done(done)
        periodic_hold = stationary and stationary_run % stationary_keep_every == 0
        keep_mask.append(structural or not stationary or periodic_hold)
        previous_action = action
        previous_state = state

    return keep_mask


def scan_episode(
    path: Path,
    *,
    action_column: str,
    state_column: str,
    done_column: str,
    thresholds: dict[str, float],
    stationary_keep_every: int,
) -> tuple[list[bool], int]:
    parquet_file = pq.ParquetFile(path)
    names = parquet_file.schema_arrow.names
    required = {action_column, state_column}
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    columns = [action_column, state_column]
    if done_column in names:
        columns.append(done_column)
    table = parquet_file.read(columns=columns)
    done_values = (
        table[done_column].to_pylist()
        if done_column in table.column_names
        else [False] * table.num_rows
    )
    mask = build_keep_mask(
        table[action_column].to_pylist(),
        table[state_column].to_pylist(),
        done_values,
        stationary_keep_every=stationary_keep_every,
        **thresholds,
    )
    return mask, table.num_rows


def _update_manifest(
    staging: Path,
    *,
    total_frames: int,
    thresholds: dict[str, float],
    stationary_keep_every: int,
) -> None:
    path = staging / "conversion_manifest.json"
    if not path.is_file():
        return
    manifest = common.read_json(path)
    manifest["total_frames"] = total_frames
    manifest["stationary_filter"] = {
        **thresholds,
        "stationary_keep_every": stationary_keep_every,
        "requires_action_and_state_stationary": True,
    }
    common.write_json(path, manifest)


def process_dataset(
    source: Path,
    output: Path,
    *,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    action_column: str,
    state_column: str,
    done_column: str,
    thresholds: dict[str, float],
    stationary_keep_every: int,
    image_stat_samples: int,
    dry_run: bool,
) -> None:
    masks: dict[int, list[bool]] = {}
    total_before = 0
    total_after = 0

    print(f"Scanning {len(episodes)} YAM episodes ...", flush=True)
    for ordinal, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        path = common.episode_path(source, info, episode_index)
        mask, length = scan_episode(
            path,
            action_column=action_column,
            state_column=state_column,
            done_column=done_column,
            thresholds=thresholds,
            stationary_keep_every=stationary_keep_every,
        )
        if length != int(episode["length"]):
            raise ValueError(f"{path}: parquet and metadata lengths differ")
        masks[episode_index] = mask
        total_before += length
        total_after += sum(mask)
        print(
            f"[{ordinal:>4}/{len(episodes)}] episode {episode_index:06d}: "
            f"{length} -> {sum(mask)}",
            flush=True,
        )

    removed = total_before - total_after
    print(
        f"Total frames: {total_before} -> {total_after}; "
        f"removed {removed} ({removed / total_before:.2%})",
        flush=True,
    )
    if dry_run:
        print("dry-run: output was not created")
        return

    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        common.copy_dataset_shell(source, staging)
        stale_norm_stats = staging / "norm_stats.json"
        if stale_norm_stats.exists():
            stale_norm_stats.unlink()
            print("Removed copied norm_stats.json; recompute it after filtering.")

        new_episodes: list[dict[str, Any]] = []
        new_stats: list[dict[str, Any]] = []
        global_index_start = 0
        fps = float(info["fps"])

        print("Writing filtered parquet files ...", flush=True)
        for ordinal, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            source_path = common.episode_path(source, info, episode_index)
            destination_path = common.episode_path(staging, info, episode_index)
            parquet_file = pq.ParquetFile(source_path)
            table = parquet_file.read()
            filtered = common.filter_and_reindex_table(
                table,
                masks[episode_index],
                episode_index=episode_index,
                global_index_start=global_index_start,
                fps=fps,
            )
            stats = common.compute_episode_stats(
                filtered,
                info=info,
                dataset_root=source,
                fps=fps,
                image_stat_samples=image_stat_samples,
            )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                filtered,
                destination_path,
                compression=common.source_compression(parquet_file),
                row_group_size=common.source_row_group_size(parquet_file),
            )

            new_episode = dict(episode)
            new_episode["length"] = filtered.num_rows
            new_episodes.append(new_episode)
            new_stats.append({"episode_index": episode_index, "stats": stats})
            global_index_start += filtered.num_rows
            print(
                f"[{ordinal:>4}/{len(episodes)}] episode {episode_index:06d} written",
                flush=True,
            )

        new_info = dict(info)
        new_info["total_frames"] = global_index_start
        meta_dir = staging / "meta"
        common.write_json(meta_dir / "info.json", new_info)
        common.write_jsonl(meta_dir / "episodes.jsonl", new_episodes)
        common.write_jsonl(meta_dir / "episodes_stats.jsonl", new_stats)
        _update_manifest(
            staging,
            total_frames=global_index_start,
            thresholds=thresholds,
            stationary_keep_every=stationary_keep_every,
        )
        common.validate_output(staging, new_info, new_episodes, new_stats)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"Finished: {output}")


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def main() -> int:
    args = parse_args()
    source = args.dataset_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(f"{source.name}_yam_stationary_filtered")
    )
    if not source.is_dir():
        raise SystemExit(f"Dataset does not exist: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit("Output must be separate from the source dataset.")
    if not args.dry_run and output.exists():
        raise SystemExit(f"Output already exists: {output}")
    if args.stationary_keep_every <= 0:
        raise SystemExit("--stationary-keep-every must be positive")
    if args.image_stat_samples <= 0:
        raise SystemExit("--image-stat-samples must be positive")

    try:
        thresholds = {
            "action_joint_epsilon": _finite_nonnegative(
                args.action_joint_epsilon, "--action-joint-epsilon"
            ),
            "action_gripper_epsilon": _finite_nonnegative(
                args.action_gripper_epsilon, "--action-gripper-epsilon"
            ),
            "state_joint_epsilon": _finite_nonnegative(
                args.state_joint_epsilon, "--state-joint-epsilon"
            ),
            "state_gripper_epsilon": _finite_nonnegative(
                args.state_gripper_epsilon, "--state-gripper-epsilon"
            ),
        }
        info, episodes = common.validate_source(source)
        process_dataset(
            source,
            output,
            info=info,
            episodes=episodes,
            action_column=args.action_column,
            state_column=args.state_column,
            done_column=args.done_column,
            thresholds=thresholds,
            stationary_keep_every=args.stationary_keep_every,
            image_stat_samples=args.image_stat_samples,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, OSError, pa.ArrowException) as exc:
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
