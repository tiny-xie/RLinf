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

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.datasets.openpi_rlinf.dual_franka.dual_franka_sft_data_loader import (
    prepare_dual_franka_online_sft_batch,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.workers.actor.fsdp_dagger_policy_worker import (
    _build_dagger_sft_forward_kwargs,
    _extract_dagger_sft_loss,
    _resolve_lerobot_preload_paths,
)
from rlinf.workers.rollout.hf.huggingface_worker import (
    merge_rollout_model_config,
)


def test_rollout_model_config_supports_nested_task_override():
    actor = OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "model_path": "actor-checkpoint",
            "precision": "fp32",
            "openpi": {
                "task": "sft",
                "config_name": "pi05_dualfranka_tcp_rot6d_state",
                "model_action_dim": 32,
            },
        }
    )
    rollout = OmegaConf.create(
        {
            "model_path": "rollout-checkpoint",
            "precision": "bf16",
            "openpi": {"task": "eval"},
        }
    )

    merged = merge_rollout_model_config(actor, rollout)

    assert merged.model_type == "openpi_rlinf"
    assert merged.model_path == "rollout-checkpoint"
    assert merged.precision == "bf16"
    assert merged.openpi.task == "eval"
    assert merged.openpi.config_name == "pi05_dualfranka_tcp_rot6d_state"
    assert merged.openpi.model_action_dim == 32
    assert actor.openpi.task == "sft"


def test_dual_franka_discrete_state_sft_config_is_registered():
    config = get_openpi_config("pi05_dualfranka_tcp_rot6d_state")
    assert config.model.discrete_state_input is True
    assert config.model.action_horizon == 20


def test_openpi_rlinf_dagger_forward_uses_structured_loss_without_legacy_kwarg():
    data = object()
    kwargs = _build_dagger_sft_forward_kwargs(SupportedModel.OPENPI_RLINF, data)

    assert kwargs == {"forward_type": ForwardType.SFT, "data": data}
    loss = torch.tensor(1.25)
    assert _extract_dagger_sft_loss({"loss": loss, "rlt_loss": loss / 2}) is loss

    with pytest.raises(ValueError, match="without a 'loss'"):
        _extract_dagger_sft_loss({"rlt_loss": loss})


def test_legacy_openpi_dagger_keeps_action_chunk_loss_kwarg():
    kwargs = _build_dagger_sft_forward_kwargs(SupportedModel.OPENPI, object())
    assert kwargs["use_action_chunk_loss"] is True


def test_resolve_lerobot_preload_paths_accepts_direct_and_rank_parent(tmp_path):
    direct = tmp_path / "direct_dataset"
    direct.mkdir()
    ranked_shard = tmp_path / "ranked" / "rank_0" / "id_3"
    ranked_shard.mkdir(parents=True)
    valid = {direct.resolve(), ranked_shard.resolve()}

    def shard_info(path):
        if path.resolve() in valid:
            return {"num_episodes": 1, "num_frames": 20}
        return None

    paths = _resolve_lerobot_preload_paths(
        [direct, tmp_path / "ranked", direct], rank=0, shard_info_fn=shard_info
    )

    assert paths == [direct, ranked_shard]


def test_resolve_lerobot_preload_paths_rejects_invalid_parent(tmp_path):
    invalid = tmp_path / "invalid"
    invalid.mkdir()

    with pytest.raises(FileNotFoundError, match="contains no valid dataset roots"):
        _resolve_lerobot_preload_paths(invalid, rank=0, shard_info_fn=lambda path: None)


def test_prepare_dual_franka_online_sft_batch_transforms_each_sample():
    batch_size = 2
    horizon = 3
    batch = {
        "image": torch.zeros(batch_size, 3, 4, 5),
        "extra_view_image-0": torch.ones(batch_size, 3, 4, 5),
        "extra_view_image-1": torch.full((batch_size, 3, 4, 5), 2.0),
        "state": torch.arange(batch_size * 20, dtype=torch.float32).reshape(
            batch_size, 20
        ),
        "actions": torch.arange(batch_size * horizon * 20, dtype=torch.float32).reshape(
            batch_size, horizon, 20
        ),
        "task": ["first task", "second task"],
    }
    seen_prompts = []

    def fake_transform(frame):
        seen_prompts.append(frame["task"])
        sample_id = len(seen_prompts) - 1
        assert frame["image"].shape == (3, 4, 5)
        padded_actions = np.pad(frame["actions"], ((0, 0), (0, 12)))
        return {
            "image": {
                "base_0_rgb": np.full((3, 2, 2), sample_id, dtype=np.float32),
                "left_wrist_0_rgb": np.zeros((3, 2, 2), dtype=np.float32),
                "right_wrist_0_rgb": np.zeros((3, 2, 2), dtype=np.float32),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.pad(frame["state"], (0, 12)),
            "tokenized_prompt": np.full(8, sample_id, dtype=np.int64),
            "tokenized_prompt_mask": np.ones(8, dtype=np.bool_),
            "actions": padded_actions,
        }

    observation, actions = prepare_dual_franka_online_sft_batch(batch, fake_transform)

    assert seen_prompts == ["first task", "second task"]
    assert observation.state.shape == (batch_size, 32)
    assert observation.images["base_0_rgb"].shape == (batch_size, 3, 2, 2)
    assert actions.shape == (batch_size, horizon, 32)
    torch.testing.assert_close(actions[..., :20], batch["actions"])
