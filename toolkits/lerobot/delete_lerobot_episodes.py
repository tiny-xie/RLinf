#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Delete indexed episodes from one LeRobot v2.1 dataset.

The tool removes episodes by their current ``episode_index``, compacts the
remaining episode and global frame indices, updates ``info.json``,
``episodes.jsonl``, and ``episodes_stats.jsonl``, and validates the rebuilt
dataset before atomically replacing the original directory.

Examples:
    Preview deletion of episode 0 from ``id_2``::

        python toolkits/lerobot/delete_lerobot_episodes.py \
            --dataset-dir /data/run/rank_0/id_2 \
            --episode-index 0 \
            --dry-run

    Delete several current indices::

        python toolkits/lerobot/delete_lerobot_episodes.py \
            --dataset-dir /data/run/rank_0/id_2 \
            --episode-index 0 5 9 \
            --yes

    Keep the original directory as a backup::

        python toolkits/lerobot/delete_lerobot_episodes.py \
            --dataset-dir /data/run/rank_0/id_2 \
            --episode-index 0 \
            --backup-dir /data/backups/id_2_before_delete \
            --yes

Only datasets with images embedded in Parquet are supported. Video-backed
datasets are rejected so that video files cannot become inconsistent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records from ``path``."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSON Lines records to ``path``."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_info(dataset_dir: Path) -> dict[str, Any]:
    """Load the dataset's ``info.json``."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    with info_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _episode_path(dataset_dir: Path, info: dict[str, Any], episode_index: int) -> Path:
    """Resolve the Parquet path for one episode."""
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    relative_path = data_path.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return dataset_dir / relative_path


def _validate_source_metadata(
    dataset_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load and validate metadata needed for rebuilding a dataset."""
    info = _load_info(dataset_dir)
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {episodes_path}")
    episodes = _read_jsonl(episodes_path)

    total_episodes = int(info.get("total_episodes", -1))
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(record["episode_index"]) for record in episodes]
    if total_episodes != len(episodes):
        raise ValueError(
            f"{dataset_dir}: info.json says {total_episodes} episodes, "
            f"but episodes.jsonl contains {len(episodes)}"
        )
    if actual_indices != expected_indices:
        raise ValueError(
            f"{dataset_dir}: source episode indices must be contiguous from 0"
        )

    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    stats = _read_jsonl(stats_path) if stats_path.is_file() else []
    if stats:
        stats_indices = [int(record["episode_index"]) for record in stats]
        if stats_indices != expected_indices:
            raise ValueError(
                f"{dataset_dir}: episodes_stats.jsonl does not match episodes.jsonl"
            )

    expected_parquets = {
        _episode_path(dataset_dir, info, episode_index).resolve()
        for episode_index in expected_indices
    }
    missing = sorted(path for path in expected_parquets if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing episode Parquet: {missing[0]}")
    actual_parquets = {
        path.resolve() for path in (dataset_dir / "data").rglob("episode_*.parquet")
    }
    unexpected = sorted(actual_parquets - expected_parquets)
    if unexpected:
        raise ValueError(f"Unexpected episode Parquet: {unexpected[0]}")

    videos_dir = dataset_dir / "videos"
    has_videos = videos_dir.is_dir() and any(
        path.is_file() for path in videos_dir.rglob("*")
    )
    if int(info.get("total_videos", 0)) or has_videos:
        raise ValueError("Video-backed LeRobot datasets are not supported by this tool")

    return info, episodes, stats


def _updated_splits(
    splits: dict[str, str], total_episodes: int, deleted: set[int]
) -> dict[str, str]:
    """Remap contiguous split ranges after removing episode indices."""
    if not splits:
        return {"train": f"0:{total_episodes - len(deleted)}"}

    updated: dict[str, str] = {}
    old_cursor = 0
    new_cursor = 0
    for name, value in splits.items():
        try:
            start_text, end_text = value.split(":", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"Unsupported split range {name}={value!r}") from error
        if start != old_cursor or not start <= end <= total_episodes:
            raise ValueError(
                "Dataset splits must be contiguous ranges covering all episodes"
            )
        retained = sum(index not in deleted for index in range(start, end))
        updated[name] = f"{new_cursor}:{new_cursor + retained}"
        old_cursor = end
        new_cursor += retained
    if old_cursor != total_episodes:
        raise ValueError("Dataset splits do not cover all episodes")
    return updated


def _source_compression(parquet_file: Any) -> str | None:
    """Return a single compression codec used by a source Parquet file."""
    metadata = parquet_file.metadata
    codecs = {
        metadata.row_group(row_group).column(column).compression.upper()
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    }
    if len(codecs) != 1:
        raise ValueError(
            "Parquet files with mixed compression codecs are not supported"
        )
    codec = codecs.pop()
    return None if codec == "UNCOMPRESSED" else codec.lower()


def _source_row_group_size(parquet_file: Any) -> int | None:
    """Infer the regular row-group size used by a source Parquet file."""
    metadata = parquet_file.metadata
    if metadata.num_row_groups == 0:
        return None
    sizes = [
        metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
    ]
    first_size = sizes[0]
    if any(size != first_size for size in sizes[:-1]) or sizes[-1] > first_size:
        raise ValueError("Irregular Parquet row groups are not supported")
    return first_size


def _rewrite_parquet(
    source: Path,
    destination: Path,
    *,
    old_episode_index: int,
    new_episode_index: int,
    new_frame_start: int,
) -> int:
    """Rewrite index columns while preserving data and schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(source)
    table = parquet_file.read()
    length = table.num_rows
    if length == 0:
        raise ValueError(f"Empty episode is not supported: {source}")

    episode_position = table.schema.get_field_index("episode_index")
    index_position = table.schema.get_field_index("index")
    if episode_position < 0 or index_position < 0:
        raise ValueError(f"Missing episode_index or index column: {source}")
    old_values = set(table.column(episode_position).to_pylist())
    if old_values != {old_episode_index}:
        raise ValueError(
            f"{source}: internal episode_index values {old_values} do not "
            f"match metadata index {old_episode_index}"
        )

    episode_field = table.schema.field(episode_position)
    index_field = table.schema.field(index_position)
    table = table.set_column(
        episode_position,
        episode_field,
        pa.array([new_episode_index] * length, type=episode_field.type),
    )
    table = table.set_column(
        index_position,
        index_field,
        pa.array(
            range(new_frame_start, new_frame_start + length),
            type=index_field.type,
        ),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    pq.write_table(
        table,
        temporary,
        compression=_source_compression(parquet_file),
        row_group_size=_source_row_group_size(parquet_file),
    )
    os.replace(temporary, destination)
    return length


def _reindex_stats(
    record: dict[str, Any],
    *,
    new_episode_index: int,
    new_frame_start: int,
    length: int,
) -> dict[str, Any]:
    """Update index-related fields in one episode statistics record."""
    updated = copy.deepcopy(record)
    updated["episode_index"] = new_episode_index
    stats = updated.get("stats", {})

    if "episode_index" in stats:
        count = stats["episode_index"].get("count", [length])
        stats["episode_index"] = {
            "min": [new_episode_index],
            "max": [new_episode_index],
            "mean": [float(new_episode_index)],
            "std": [0.0],
            "count": count,
        }
    if "index" in stats:
        old_index_stats = stats["index"]
        stats["index"] = {
            "min": [new_frame_start],
            "max": [new_frame_start + length - 1],
            "mean": [new_frame_start + (length - 1) / 2],
            "std": old_index_stats.get("std", [0.0]),
            "count": old_index_stats.get("count", [length]),
        }
    return updated


def _copy_non_data_files(source: Path, destination: Path) -> None:
    """Copy metadata and auxiliary files, excluding episode data."""
    destination.mkdir(parents=True)
    for child in source.iterdir():
        if child.name in {"data", "videos"}:
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _validate_rebuilt_dataset(dataset_dir: Path) -> None:
    """Validate metadata, paths, row counts, and internal indices."""
    import pyarrow.parquet as pq

    info, episodes, stats = _validate_source_metadata(dataset_dir)
    expected_indices = list(range(len(episodes)))
    if stats and [record["episode_index"] for record in stats] != expected_indices:
        raise ValueError("Rebuilt statistics indices are not contiguous")

    next_frame_index = 0
    for record in episodes:
        episode_index = int(record["episode_index"])
        parquet_path = _episode_path(dataset_dir, info, episode_index)
        table = pq.read_table(parquet_path, columns=["episode_index", "index"])
        length = table.num_rows
        if length != int(record["length"]):
            raise ValueError(f"{parquet_path}: row count does not match metadata")
        if set(table.column("episode_index").to_pylist()) != {episode_index}:
            raise ValueError(f"{parquet_path}: internal episode_index mismatch")
        expected_frames = list(range(next_frame_index, next_frame_index + length))
        if table.column("index").to_pylist() != expected_frames:
            raise ValueError(f"{parquet_path}: internal global index mismatch")

        if stats:
            episode_stats = stats[episode_index].get("stats", {})
            if "episode_index" in episode_stats:
                if episode_stats["episode_index"].get("min") != [episode_index]:
                    raise ValueError("Statistics episode_index mismatch")
            if "index" in episode_stats:
                if episode_stats["index"].get("min") != [next_frame_index]:
                    raise ValueError("Statistics global index mismatch")
        next_frame_index += length

    if next_frame_index != int(info["total_frames"]):
        raise ValueError("Rebuilt total_frames does not match episode data")


def _build_staged_dataset(
    source: Path,
    destination: Path,
    *,
    deleted_indices: set[int],
) -> dict[str, Any]:
    """Build and validate a filtered dataset in ``destination``."""
    info, episodes, stats = _validate_source_metadata(source)
    stats_by_index = {int(record["episode_index"]): record for record in stats}
    _copy_non_data_files(source, destination)

    new_episodes: list[dict[str, Any]] = []
    new_stats: list[dict[str, Any]] = []
    mapping: dict[int, int] = {}
    frame_start = 0
    removed_frames = 0
    for old_record in episodes:
        old_episode_index = int(old_record["episode_index"])
        if old_episode_index in deleted_indices:
            removed_frames += int(old_record["length"])
            continue

        new_episode_index = len(new_episodes)
        source_path = _episode_path(source, info, old_episode_index)
        destination_path = _episode_path(destination, info, new_episode_index)
        length = _rewrite_parquet(
            source_path,
            destination_path,
            old_episode_index=old_episode_index,
            new_episode_index=new_episode_index,
            new_frame_start=frame_start,
        )
        if length != int(old_record["length"]):
            raise ValueError(
                f"{source_path}: row count {length} does not match metadata "
                f"length {old_record['length']}"
            )

        new_record = copy.deepcopy(old_record)
        new_record["episode_index"] = new_episode_index
        new_record["length"] = length
        new_episodes.append(new_record)
        mapping[old_episode_index] = new_episode_index
        if stats:
            new_stats.append(
                _reindex_stats(
                    stats_by_index[old_episode_index],
                    new_episode_index=new_episode_index,
                    new_frame_start=frame_start,
                    length=length,
                )
            )
        frame_start += length

    old_total = len(episodes)
    new_total = len(new_episodes)
    info["total_episodes"] = new_total
    info["total_frames"] = frame_start
    info["total_chunks"] = max(
        1, math.ceil(new_total / int(info.get("chunks_size", 1000)))
    )
    info["splits"] = _updated_splits(info.get("splits", {}), old_total, deleted_indices)

    meta_dir = destination / "meta"
    with (meta_dir / "info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    _write_jsonl(meta_dir / "episodes.jsonl", new_episodes)
    if stats:
        _write_jsonl(meta_dir / "episodes_stats.jsonl", new_stats)

    _validate_rebuilt_dataset(destination)
    return {
        "dataset_dir": str(source),
        "deleted_indices": sorted(deleted_indices),
        "removed_frames": removed_frames,
        "before_episodes": old_total,
        "after_episodes": new_total,
        "before_frames": int(info["total_frames"]) + removed_frames,
        "after_frames": frame_start,
        "index_mapping": mapping,
    }


def _preview(
    dataset_dir: Path,
    episodes: list[dict[str, Any]],
    deleted_indices: set[int],
) -> dict[str, Any]:
    """Build a deletion preview without reading or writing Parquet content."""
    removed = [episodes[index] for index in sorted(deleted_indices)]
    return {
        "dataset_dir": str(dataset_dir),
        "deleted_indices": sorted(deleted_indices),
        "removed_frames": sum(int(record["length"]) for record in removed),
        "before_episodes": len(episodes),
        "after_episodes": len(episodes) - len(deleted_indices),
        "before_frames": sum(int(record["length"]) for record in episodes),
        "after_frames": sum(
            int(record["length"])
            for record in episodes
            if int(record["episode_index"]) not in deleted_indices
        ),
    }


def delete_episodes(
    dataset_dir: str | Path,
    episode_indices: set[int],
    *,
    dry_run: bool = False,
    backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Delete indexed episodes and atomically replace a LeRobot dataset.

    Args:
        dataset_dir: Path to one LeRobot dataset, such as ``rank_0/id_2``.
        episode_indices: Current episode indices to delete.
        dry_run: Only validate metadata and return a preview when true.
        backup_dir: Optional path at which to retain the original dataset.

    Returns:
        A dictionary describing the deletion and resulting counts.

    Raises:
        ValueError: If indices or dataset metadata are invalid.
        FileNotFoundError: If required dataset files are absent.
    """
    source = Path(dataset_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {source}")
    if not episode_indices:
        raise ValueError("At least one episode index is required")
    if any(index < 0 for index in episode_indices):
        raise ValueError("Episode indices must be non-negative")

    _, episodes, _ = _validate_source_metadata(source)
    available = set(range(len(episodes)))
    missing = sorted(episode_indices - available)
    if missing:
        raise ValueError(f"Episode indices do not exist: {missing}")
    if episode_indices == available:
        raise ValueError("Refusing to delete every episode in the dataset")
    if dry_run:
        return _preview(source, episodes, episode_indices)

    retained_backup = (
        Path(backup_dir).expanduser().resolve() if backup_dir is not None else None
    )
    if retained_backup is not None:
        if retained_backup.exists():
            raise FileExistsError(f"Backup path already exists: {retained_backup}")
        if source in retained_backup.parents or retained_backup == source:
            raise ValueError("Backup directory cannot be inside the dataset")
        if retained_backup.parent.stat().st_dev != source.parent.stat().st_dev:
            raise ValueError("Backup directory must be on the same filesystem")

    work_dir = source.parent / f".delete_episodes_work_{uuid.uuid4().hex}"
    staged = work_dir / "staged"
    temporary_backup = work_dir / "original"
    work_dir.mkdir()
    original_location = retained_backup or temporary_backup
    replaced = False
    try:
        report = _build_staged_dataset(source, staged, deleted_indices=episode_indices)
        if retained_backup is not None:
            retained_backup.parent.mkdir(parents=True, exist_ok=True)

        os.replace(source, original_location)
        try:
            os.replace(staged, source)
            replaced = True
            _validate_rebuilt_dataset(source)
        except Exception:
            if source.exists():
                os.replace(source, staged)
            os.replace(original_location, source)
            raise
        return report
    finally:
        if work_dir.exists():
            if not replaced and temporary_backup.exists() and not source.exists():
                os.replace(temporary_backup, source)
            shutil.rmtree(work_dir)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Delete current episode indices from one LeRobot dataset and "
            "consistently rebuild its metadata."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="LeRobot dataset directory, for example rank_0/id_2.",
    )
    parser.add_argument(
        "--episode-index",
        required=True,
        type=int,
        nargs="+",
        help="One or more current episode indices to delete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned deletion without modifying files.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Keep the complete original dataset at this path.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm permanent deletion; not needed with --dry-run.",
    )
    return parser


def main() -> None:
    """Run the command-line interface."""
    args = _build_parser().parse_args()
    if not args.dry_run and not args.yes:
        print(
            "Refusing to modify data without --yes. Use --dry-run to preview.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        report = delete_episodes(
            args.dataset_dir,
            set(args.episode_index),
            dry_run=args.dry_run,
            backup_dir=args.backup_dir,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    action = "Would delete" if args.dry_run else "Deleted"
    print(f"{action} episode indices: {report['deleted_indices']}")
    print(
        "Episodes: "
        f"{report['before_episodes']} -> {report['after_episodes']}; "
        "frames: "
        f"{report['before_frames']} -> {report['after_frames']}"
    )
    if args.backup_dir is not None and not args.dry_run:
        print(f"Original dataset retained at: {args.backup_dir.resolve()}")


if __name__ == "__main__":
    main()
