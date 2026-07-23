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

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_tool():
    tool_path = (
        Path(__file__).resolve().parents[2]
        / "toolkits"
        / "lerobot"
        / "delete_lerobot_episodes.py"
    )
    spec = importlib.util.spec_from_file_location("delete_lerobot_episodes", tool_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _make_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    lengths = [2, 3, 1, 4]
    frame_start = 0
    episodes = []
    episode_stats = []
    for episode_index, length in enumerate(lengths):
        table = pa.table(
            {
                "value": pa.array(range(length), type=pa.int64()),
                "frame_index": pa.array(range(length), type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "index": pa.array(
                    range(frame_start, frame_start + length), type=pa.int64()
                ),
            }
        )
        pq.write_table(
            table,
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet",
            compression="snappy",
            row_group_size=2,
        )
        episodes.append(
            {"episode_index": episode_index, "tasks": ["test"], "length": length}
        )
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "episode_index": {
                        "min": [episode_index],
                        "max": [episode_index],
                        "mean": [float(episode_index)],
                        "std": [0.0],
                        "count": [length],
                    },
                    "index": {
                        "min": [frame_start],
                        "max": [frame_start + length - 1],
                        "mean": [frame_start + (length - 1) / 2],
                        "std": [0.5],
                        "count": [length],
                    },
                },
            }
        )
        frame_start += length

    info = {
        "codebase_version": "v2.1",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
    }
    with (root / "meta" / "info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle)
    _write_jsonl(root / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(root / "meta" / "episodes_stats.jsonl", episode_stats)


def test_dry_run_does_not_modify_dataset(tmp_path):
    dataset = tmp_path / "id_2"
    _make_dataset(dataset)

    report = TOOL.delete_episodes(dataset, {0, 2}, dry_run=True)

    assert report["deleted_indices"] == [0, 2]
    assert report["before_episodes"] == 4
    assert report["after_episodes"] == 2
    assert report["before_frames"] == 10
    assert report["after_frames"] == 7
    assert len(TOOL._read_jsonl(dataset / "meta" / "episodes.jsonl")) == 4


def test_delete_reindexes_data_and_metadata(tmp_path):
    dataset = tmp_path / "id_2"
    _make_dataset(dataset)

    report = TOOL.delete_episodes(dataset, {0, 2})

    assert report["deleted_indices"] == [0, 2]
    assert report["after_episodes"] == 2
    assert report["after_frames"] == 7
    info = TOOL._load_info(dataset)
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 7
    assert info["splits"] == {"train": "0:2"}

    episodes = TOOL._read_jsonl(dataset / "meta" / "episodes.jsonl")
    stats = TOOL._read_jsonl(dataset / "meta" / "episodes_stats.jsonl")
    assert [record["episode_index"] for record in episodes] == [0, 1]
    assert [record["length"] for record in episodes] == [3, 4]
    assert [record["episode_index"] for record in stats] == [0, 1]
    assert stats[1]["stats"]["index"]["min"] == [3]
    assert stats[1]["stats"]["index"]["max"] == [6]

    first = pq.read_table(dataset / "data" / "chunk-000" / "episode_000000.parquet")
    second = pq.read_table(dataset / "data" / "chunk-000" / "episode_000001.parquet")
    assert first.column("episode_index").to_pylist() == [0, 0, 0]
    assert first.column("index").to_pylist() == [0, 1, 2]
    assert second.column("episode_index").to_pylist() == [1, 1, 1, 1]
    assert second.column("index").to_pylist() == [3, 4, 5, 6]


def test_invalid_index_leaves_dataset_unchanged(tmp_path):
    dataset = tmp_path / "id_2"
    _make_dataset(dataset)

    with pytest.raises(ValueError, match="do not exist"):
        TOOL.delete_episodes(dataset, {99})

    assert TOOL._load_info(dataset)["total_episodes"] == 4
    assert not list(tmp_path.glob(".delete_episodes_work_*"))
