#!/usr/bin/env python3
"""Filter stationary frames from a complete LeRobot v2.1 dataset.

The input must be a dataset root containing ``data/`` and ``meta/``. The
script creates a new dataset, filters every episode Parquet file, compacts the
frame/global indices, and rebuilds the metadata statistics.

Example:
    python3 filter_zero_action_frames.py /path/to/dataset \
        -o /path/to/dataset_filtered --mode delta --epsilon 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
except ImportError as exc:  # pragma: no cover - only runs without dependencies
    raise SystemExit(
        "缺少依赖，请运行：python3 -m pip install numpy pillow pyarrow"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "滤除 LeRobot v2.1 数据集中所有 episode 的零/静止动作帧，"
            "并同步重建 meta。"
        )
    )
    parser.add_argument(
        "dataset_root", type=Path, help="包含 data/ 和 meta/ 的数据集根目录"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="新数据集目录（默认：在输入目录旁添加 _no_zero_actions）",
    )
    parser.add_argument(
        "--action-column", default="actions", help="动作列名（默认：actions）"
    )
    parser.add_argument(
        "--mode",
        choices=("value", "delta"),
        default="delta",
        help=(
            "delta：动作相对上一源帧几乎不变（默认，适合绝对控制量）；"
            "value：动作向量本身接近全零"
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="所有动作维度均不超过该阈值时判为零/静止（默认：1e-3）",
    )
    parser.add_argument(
        "--ignore-indices",
        default="",
        help="判零时忽略的动作维度，例如 '9,19'（默认：不忽略）",
    )
    parser.add_argument(
        "--done-column", default="done", help="终止标记列名（默认：done）"
    )
    parser.add_argument(
        "--image-stat-samples",
        type=int,
        default=1,
        help="每个 episode 重算图像统计时均匀抽样的帧数（默认：100）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描动作列并报告过滤数量，不创建输出数据集",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=4, allow_nan=False)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )


def parse_ignored_indices(text: str) -> set[int]:
    if not text.strip():
        return set()
    try:
        indices = {int(part.strip()) for part in text.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("--ignore-indices 必须是逗号分隔的整数，例如 9,19") from exc
    if any(index < 0 for index in indices):
        raise ValueError("--ignore-indices 不接受负数下标")
    return indices


def is_near_zero(values: Sequence[object], epsilon: float, ignored: set[int]) -> bool:
    checked = 0
    for index, value in enumerate(values):
        if index in ignored:
            continue
        checked += 1
        if value is None:
            return False
        number = float(value)
        if not math.isfinite(number) or abs(number) > epsilon:
            return False
    if checked == 0:
        raise ValueError("所有动作维度都被忽略了，无法判断零动作")
    return True


def is_near_zero_delta(
    current: Sequence[object],
    previous: Sequence[object],
    epsilon: float,
    ignored: set[int],
) -> bool:
    if len(current) != len(previous):
        return False
    checked = 0
    for index, (current_value, previous_value) in enumerate(zip(current, previous)):
        if index in ignored:
            continue
        checked += 1
        if current_value is None or previous_value is None:
            return False
        current_number = float(current_value)
        previous_number = float(previous_value)
        if not math.isfinite(current_number) or not math.isfinite(previous_number):
            return False
        if abs(current_number - previous_number) > epsilon:
            return False
    if checked == 0:
        raise ValueError("所有动作维度都被忽略了，无法判断零动作")
    return True


def build_keep_mask(
    actions: Sequence[Sequence[object] | None],
    done_values: Sequence[object],
    *,
    mode: str,
    epsilon: float,
    ignored: set[int],
) -> list[bool]:
    """Return a mask; the first and terminal source frames are always retained."""
    keep_mask: list[bool] = []
    previous: Sequence[object] | None = None
    last_index = len(actions) - 1

    for index, (action, done) in enumerate(zip(actions, done_values)):
        zero_action = False
        if action is not None:
            if ignored and max(ignored) >= len(action):
                raise ValueError(
                    f"忽略维度 {max(ignored)} 超出动作向量长度 {len(action)}"
                )
            if mode == "value":
                zero_action = is_near_zero(action, epsilon, ignored)
            elif previous is not None:
                zero_action = is_near_zero_delta(action, previous, epsilon, ignored)
            # Delta is measured against the preceding source frame, not the
            # preceding retained frame, so slow movement is not accumulated.
            previous = action
        else:
            previous = None

        is_structural_frame = index == 0 or index == last_index or bool(done)
        keep_mask.append(not zero_action or is_structural_frame)
    return keep_mask


def episode_path(
    dataset_root: Path, info: dict[str, Any], episode_index: int
) -> Path:
    chunks_size = int(info.get("chunks_size", 1000))
    path_template = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    try:
        relative = path_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"不支持的 info.json data_path：{path_template!r}") from exc
    return dataset_root / relative


def validate_source(
    dataset_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data_dir = dataset_root / "data"
    meta_dir = dataset_root / "meta"
    if not data_dir.is_dir() or not meta_dir.is_dir():
        raise FileNotFoundError(
            f"{dataset_root} 不是完整数据集根目录：必须同时包含 data/ 和 meta/"
        )

    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError("meta/ 中缺少 info.json 或 episodes.jsonl")

    info = read_json(info_path)
    if info.get("codebase_version") != "v2.1":
        raise ValueError(
            "当前脚本只支持 LeRobot v2.1，检测到版本："
            f"{info.get('codebase_version')!r}"
        )
    episodes = read_jsonl(episodes_path)
    expected_indices = list(range(len(episodes)))
    actual_indices = [int(record["episode_index"]) for record in episodes]
    if actual_indices != expected_indices:
        raise ValueError("episodes.jsonl 的 episode_index 必须从 0 连续递增")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json 的 total_episodes 与 episodes.jsonl 不一致")

    expected_paths = {
        episode_path(dataset_root, info, index).resolve()
        for index in expected_indices
    }
    missing = sorted(path for path in expected_paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"缺少 episode 文件：{missing[0]}")
    actual_paths = {
        path.resolve() for path in data_dir.rglob("episode_*.parquet")
    }
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise ValueError(f"data/ 中存在未登记的 episode 文件：{unexpected[0]}")

    videos_dir = dataset_root / "videos"
    has_video_files = videos_dir.is_dir() and any(
        path.is_file() for path in videos_dir.rglob("*")
    )
    if int(info.get("total_videos", 0)) or has_video_files:
        raise ValueError("暂不支持视频型数据集：删帧后还需要同步重编码视频")
    return info, episodes


def source_compression(parquet_file: pq.ParquetFile) -> str | None:
    codecs = {
        parquet_file.metadata.row_group(row_group).column(column).compression.upper()
        for row_group in range(parquet_file.metadata.num_row_groups)
        for column in range(parquet_file.metadata.num_columns)
    }
    if len(codecs) != 1:
        raise ValueError("暂不支持一个 Parquet 内混用多种压缩格式")
    codec = codecs.pop()
    return None if codec == "UNCOMPRESSED" else codec.lower()


def source_row_group_size(parquet_file: pq.ParquetFile) -> int | None:
    metadata = parquet_file.metadata
    if metadata.num_row_groups == 0:
        return None
    sizes = [
        metadata.row_group(index).num_rows
        for index in range(metadata.num_row_groups)
    ]
    first = sizes[0]
    if any(size != first for size in sizes[:-1]) or sizes[-1] > first:
        return None
    return first


def replace_column(table: pa.Table, name: str, values: Sequence[object]) -> pa.Table:
    position = table.schema.get_field_index(name)
    if position < 0:
        raise ValueError(f"Parquet 缺少必要列：{name}")
    field = table.schema.field(position)
    return table.set_column(position, field, pa.array(values, type=field.type))


def filter_and_reindex_table(
    table: pa.Table,
    keep_mask: Sequence[bool],
    *,
    episode_index: int,
    global_index_start: int,
    fps: float,
) -> pa.Table:
    filtered = table.filter(pa.array(keep_mask, type=pa.bool_()))
    length = filtered.num_rows
    if length == 0:
        raise ValueError(f"episode {episode_index} 过滤后为空")

    filtered = replace_column(filtered, "frame_index", range(length))
    filtered = replace_column(
        filtered, "index", range(global_index_start, global_index_start + length)
    )
    filtered = replace_column(filtered, "episode_index", [episode_index] * length)
    filtered = replace_column(
        filtered, "timestamp", [frame / fps for frame in range(length)]
    )
    return filtered


def numeric_values(column: pa.ChunkedArray, name: str) -> np.ndarray:
    array = column.combine_chunks()
    if array.null_count:
        raise ValueError(f"列 {name!r} 含有空值，无法生成统计")

    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        rows = array.to_pylist()
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError(f"列 {name!r} 的向量长度不一致")
        dtype = array.type.value_type.to_pandas_dtype()
        return np.asarray(rows, dtype=dtype)
    if pa.types.is_fixed_size_list(array.type):
        dtype = array.type.value_type.to_pandas_dtype()
        return np.asarray(array.to_pylist(), dtype=dtype)
    if (
        pa.types.is_boolean(array.type)
        or pa.types.is_integer(array.type)
        or pa.types.is_floating(array.type)
    ):
        return np.asarray(array.to_numpy(zero_copy_only=False)).reshape(-1, 1)
    raise ValueError(f"列 {name!r} 的类型 {array.type} 不支持统计")


def stats_for_array(values: np.ndarray, name: str) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError(f"列 {name!r} 为空，无法生成统计")
    if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
        raise ValueError(f"列 {name!r} 含有 NaN/Inf，无法生成统计")
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def image_from_record(record: dict[str, Any], dataset_root: Path) -> Image.Image:
    image_bytes = record.get("bytes")
    if image_bytes is not None:
        return Image.open(BytesIO(image_bytes))
    image_path = record.get("path")
    if not image_path:
        raise ValueError("图像记录既没有 bytes 也没有 path")
    path = Path(image_path)
    if not path.is_absolute():
        path = dataset_root / path
    return Image.open(path)


def image_stats(
    column: pa.ChunkedArray,
    *,
    dataset_root: Path,
    channels: int,
    max_samples: int,
    name: str,
) -> dict[str, Any]:
    array = column.combine_chunks()
    length = len(array)
    sample_count = min(length, max_samples)
    # This matches LeRobot v2.1: rounded, evenly spaced sample positions.
    sample_indices = np.round(np.linspace(0, length - 1, sample_count)).astype(int)

    channel_min = np.full(channels, np.inf, dtype=np.float64)
    channel_max = np.full(channels, -np.inf, dtype=np.float64)
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_square_sum = np.zeros(channels, dtype=np.float64)
    pixels_per_channel = 0

    for index in sample_indices:
        record = array[int(index)].as_py()
        if record is None:
            raise ValueError(f"图像列 {name!r} 含有空值")
        with image_from_record(record, dataset_root) as image:
            if channels == 1:
                pixels = np.asarray(image.convert("L"), dtype=np.float64)[..., None]
            elif channels == 3:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
            elif channels == 4:
                pixels = np.asarray(image.convert("RGBA"), dtype=np.float64)
            else:
                raise ValueError(f"图像列 {name!r} 的通道数 {channels} 不受支持")
        pixels /= 255.0
        flat = pixels.reshape(-1, channels)
        channel_min = np.minimum(channel_min, flat.min(axis=0))
        channel_max = np.maximum(channel_max, flat.max(axis=0))
        channel_sum += flat.sum(axis=0)
        channel_square_sum += np.square(flat).sum(axis=0)
        pixels_per_channel += flat.shape[0]

    mean = channel_sum / pixels_per_channel
    variance = np.maximum(channel_square_sum / pixels_per_channel - np.square(mean), 0)
    std = np.sqrt(variance)

    def nested(values: np.ndarray) -> list[list[list[float]]]:
        return values.reshape(channels, 1, 1).tolist()

    return {
        "min": nested(channel_min),
        "max": nested(channel_max),
        "mean": nested(mean),
        "std": nested(std),
        "count": [sample_count],
    }


def compute_episode_stats(
    table: pa.Table,
    *,
    info: dict[str, Any],
    dataset_root: Path,
    fps: float,
    image_stat_samples: int,
) -> dict[str, Any]:
    features = info.get("features", {})
    stats: dict[str, Any] = {}
    for name in table.column_names:
        feature = features.get(name, {})
        if feature.get("dtype") == "image":
            shape = feature.get("shape", [])
            if len(shape) != 3:
                raise ValueError(f"图像列 {name!r} 的 shape 无效：{shape!r}")
            stats[name] = image_stats(
                table[name],
                dataset_root=dataset_root,
                channels=int(shape[-1]),
                max_samples=image_stat_samples,
                name=name,
            )
        elif name == "timestamp":
            # LeRobot creates timestamps as Python float before casting Parquet
            # to float32; use the same float64 values for its metadata stats.
            timestamps = (np.arange(table.num_rows, dtype=np.float64) / fps).reshape(
                -1, 1
            )
            stats[name] = stats_for_array(timestamps, name)
        else:
            stats[name] = stats_for_array(numeric_values(table[name], name), name)
    return stats


def copy_dataset_shell(source: Path, destination: Path) -> None:
    """Copy metadata and non-data files into a new staging directory."""
    destination.mkdir()
    shutil.copytree(source / "meta", destination / "meta")
    (destination / "data").mkdir()
    for child in source.iterdir():
        if child.name in {"data", "meta"}:
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def validate_output(
    dataset_root: Path,
    info: dict[str, Any],
    episodes: Sequence[dict[str, Any]],
    episode_stats_records: Sequence[dict[str, Any]],
) -> None:
    if int(info["total_frames"]) != sum(int(record["length"]) for record in episodes):
        raise ValueError("输出 info.json total_frames 与 episodes.jsonl 不一致")
    if len(episode_stats_records) != len(episodes):
        raise ValueError("输出 episodes_stats.jsonl 条数不正确")

    global_start = 0
    fps = float(info["fps"])
    for episode, stats_record in zip(episodes, episode_stats_records):
        episode_index = int(episode["episode_index"])
        path = episode_path(dataset_root, info, episode_index)
        table = pq.read_table(
            path,
            columns=["frame_index", "episode_index", "index", "timestamp"],
        )
        length = int(episode["length"])
        if table.num_rows != length:
            raise ValueError(f"{path}: 行数与 episodes.jsonl 不一致")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"{path}: frame_index 不连续")
        if set(table["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"{path}: episode_index 不正确")
        expected_global = list(range(global_start, global_start + length))
        if table["index"].to_pylist() != expected_global:
            raise ValueError(f"{path}: 全局 index 不连续")
        actual_timestamps = np.asarray(table["timestamp"].to_pylist())
        expected_timestamps = np.arange(length, dtype=np.float64) / fps
        if not np.allclose(actual_timestamps, expected_timestamps, atol=1e-6):
            raise ValueError(f"{path}: timestamp 不正确")
        if int(stats_record["episode_index"]) != episode_index:
            raise ValueError("episodes_stats.jsonl 的 episode_index 不正确")
        for feature_stats in stats_record["stats"].values():
            count = feature_stats.get("count")
            if not isinstance(count, list) or not count or int(count[0]) > length:
                raise ValueError("episodes_stats.jsonl 中存在无效 count")
        global_start += length


def scan_episode(
    path: Path,
    *,
    action_column: str,
    done_column: str,
    mode: str,
    epsilon: float,
    ignored: set[int],
) -> tuple[list[bool], int]:
    parquet_file = pq.ParquetFile(path)
    names = parquet_file.schema_arrow.names
    if action_column not in names:
        raise ValueError(f"{path}: 找不到动作列 {action_column!r}")
    columns = [action_column]
    if done_column in names:
        columns.append(done_column)
    table = parquet_file.read(columns=columns)
    actions = table[action_column].to_pylist()
    done_values = (
        table[done_column].to_pylist()
        if done_column in table.column_names
        else [False] * table.num_rows
    )
    mask = build_keep_mask(
        actions,
        done_values,
        mode=mode,
        epsilon=epsilon,
        ignored=ignored,
    )
    return mask, table.num_rows


def process_dataset(
    source: Path,
    output: Path,
    *,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    action_column: str,
    done_column: str,
    mode: str,
    epsilon: float,
    ignored: set[int],
    image_stat_samples: int,
    dry_run: bool,
) -> None:
    total_before = 0
    total_after = 0
    masks: dict[int, list[bool]] = {}

    print(f"扫描 {len(episodes)} 个 episodes ...")
    for ordinal, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        path = episode_path(source, info, episode_index)
        mask, length = scan_episode(
            path,
            action_column=action_column,
            done_column=done_column,
            mode=mode,
            epsilon=epsilon,
            ignored=ignored,
        )
        metadata_length = int(episode["length"])
        if length != metadata_length:
            raise ValueError(
                f"{path}: Parquet 行数 {length} 与 meta 长度 {metadata_length} 不一致"
            )
        kept = sum(mask)
        masks[episode_index] = mask
        total_before += length
        total_after += kept
        print(
            f"[{ordinal:>4}/{len(episodes)}] episode {episode_index:06d}: "
            f"{length} -> {kept}（滤除 {length - kept}）"
        )

    removed = total_before - total_after
    print(f"总帧数：{total_before} -> {total_after}")
    print(f"预计滤除：{removed}（{removed / total_before:.2%}）")
    if dry_run:
        print("dry-run：未创建输出数据集")
        return

    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        copy_dataset_shell(source, staging)
        new_episodes: list[dict[str, Any]] = []
        new_stats: list[dict[str, Any]] = []
        global_index_start = 0
        fps = float(info["fps"])

        print("写入过滤后的 Parquet，并重算 episode 统计 ...")
        for ordinal, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            source_path = episode_path(source, info, episode_index)
            destination_path = episode_path(staging, info, episode_index)
            parquet_file = pq.ParquetFile(source_path)
            table = parquet_file.read()
            filtered = filter_and_reindex_table(
                table,
                masks[episode_index],
                episode_index=episode_index,
                global_index_start=global_index_start,
                fps=fps,
            )

            stats = compute_episode_stats(
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
                compression=source_compression(parquet_file),
                row_group_size=source_row_group_size(parquet_file),
            )

            new_episode = dict(episode)
            new_episode["length"] = filtered.num_rows
            new_episodes.append(new_episode)
            new_stats.append({"episode_index": episode_index, "stats": stats})
            global_index_start += filtered.num_rows
            print(
                f"[{ordinal:>4}/{len(episodes)}] episode {episode_index:06d} 完成"
            )

        new_info = dict(info)
        new_info["total_frames"] = global_index_start
        meta_dir = staging / "meta"
        write_json(meta_dir / "info.json", new_info)
        write_jsonl(meta_dir / "episodes.jsonl", new_episodes)
        write_jsonl(meta_dir / "episodes_stats.jsonl", new_stats)

        print("验证输出索引和元数据 ...")
        validate_output(staging, new_info, new_episodes, new_stats)
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(f"处理完成：{output}")


def main() -> int:
    args = parse_args()
    source = args.dataset_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(f"{source.name}_no_zero_actions")
    )
    if not source.is_dir():
        raise SystemExit(f"数据集目录不存在：{source}")
    if source == output or source in output.parents or output in source.parents:
        raise SystemExit("输出目录不能等于、包含或位于输入数据集目录内")
    if not args.dry_run and output.exists():
        raise SystemExit(f"输出目录已存在，为避免覆盖已停止：{output}")
    if args.epsilon < 0 or not math.isfinite(args.epsilon):
        raise SystemExit("--epsilon 必须是有限的非负数")
    if args.image_stat_samples <= 0:
        raise SystemExit("--image-stat-samples 必须大于 0")

    try:
        ignored = parse_ignored_indices(args.ignore_indices)
        info, episodes = validate_source(source)
        process_dataset(
            source,
            output,
            info=info,
            episodes=episodes,
            action_column=args.action_column,
            done_column=args.done_column,
            mode=args.mode,
            epsilon=args.epsilon,
            ignored=ignored,
            image_stat_samples=args.image_stat_samples,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise SystemExit(f"错误：{exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
