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

import json

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from rlinf.data.datasets.dagger import RollingLeRobotDataset


def _write_archived_shard(root, num_frames: int = 4):
    data_dir = root / "data" / "chunk-000"
    meta_dir = root / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()

    features = Features(
        {
            "state": Sequence(Value("float32"), length=2),
            "actions": Sequence(Value("float32"), length=2),
            "intervene_flag": Value("bool"),
            "image": Image(),
            "timestamp": Value("float32"),
            "frame_index": Value("int64"),
            "episode_index": Value("int64"),
            "index": Value("int64"),
            "task_index": Value("int64"),
        }
    )
    dataset = Dataset.from_dict(
        {
            "state": np.arange(num_frames * 2, dtype=np.float32).reshape(num_frames, 2),
            "actions": np.arange(num_frames * 2, dtype=np.float32).reshape(
                num_frames, 2
            ),
            "intervene_flag": [True] * num_frames,
            "image": [
                PILImage.fromarray(np.full((4, 5, 3), i, dtype=np.uint8))
                for i in range(num_frames)
            ],
            "timestamp": np.arange(num_frames, dtype=np.float32) / 10,
            "frame_index": np.arange(num_frames, dtype=np.int64),
            "episode_index": np.zeros(num_frames, dtype=np.int64),
            "index": np.arange(num_frames, dtype=np.int64),
            "task_index": np.zeros(num_frames, dtype=np.int64),
        },
        features=features,
    )
    pq.write_table(dataset.data.table, data_dir / "episode_000000.parquet")

    info = {
        "total_episodes": 1,
        "total_frames": num_frames,
        "chunks_size": 1000,
        "fps": 10,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "episodes.jsonl").write_text(
        json.dumps(
            {"episode_index": 0, "tasks": ["fast preload"], "length": num_frames}
        )
        + "\n"
    )
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "fast preload"}) + "\n"
    )


def test_archived_preload_uses_lazy_arrow_image_decode(tmp_path, monkeypatch):
    shard = tmp_path / "offline"
    _write_archived_shard(shard)

    def fail_eager_decode(*args, **kwargs):
        raise AssertionError("archived preload must not eagerly decode image rows")

    monkeypatch.setattr(
        "rlinf.data.datasets.dagger.dataset._load_lerobot_episode_frames",
        fail_eager_decode,
    )
    dataset = RollingLeRobotDataset(
        root_dir=tmp_path / "online",
        chunk_size=3,
        min_frames=1,
        require_all_intervene=True,
        window_size=100,
        in_memory_mode=True,
        fps=10,
    )

    staged = dataset.load_archived_shards_staged([shard])
    assert dataset.publish_staged_resume_shards(staged) == (1, 4)
    assert dataset.get_stats()["logical_samples"] == 4

    item = dataset[3]
    assert item["task"] == "fast preload"
    assert item["image"].shape == (3, 4, 5)
    assert item["image"].dtype == torch.uint8
    assert item["actions"].shape == (3, 2)
    assert item["actions_is_pad"].tolist() == [False, True, True]
