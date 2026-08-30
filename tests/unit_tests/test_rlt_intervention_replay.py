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

import inspect
from types import SimpleNamespace

import torch

from rlinf.algorithms.rlt.replay import RLTRawReplayBuilder
from rlinf.algorithms.rlt.transition import (
    filter_intervention_replay_trajectories,
    use_intervention_only_replay,
)
from rlinf.data.embodied_io_struct import Trajectory


def _trajectory_with_one_intervention_window() -> Trajectory:
    actions = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8)
    intervene_flags = torch.zeros_like(actions, dtype=torch.bool)
    intervene_flags[1, 0, 2] = True
    return Trajectory(
        max_episode_length=3,
        actions=actions,
        intervene_flags=intervene_flags,
        rewards=torch.arange(12, dtype=torch.float32).reshape(3, 1, 4),
        terminations=torch.zeros((3, 1, 4), dtype=torch.bool),
        truncations=torch.zeros((3, 1, 4), dtype=torch.bool),
        dones=torch.zeros((3, 1, 4), dtype=torch.bool),
        curr_obs={"z_rl": torch.arange(6).reshape(3, 1, 2)},
        next_obs={"z_rl": torch.arange(6, 12).reshape(3, 1, 2)},
    )


def test_intervention_only_replay_config_defaults_off():
    cfg = SimpleNamespace(algorithm={"replay_buffer": {}})
    assert not use_intervention_only_replay(cfg)

    cfg.algorithm["replay_buffer"]["rlt_intervention_only"] = True
    assert use_intervention_only_replay(cfg)


def test_filter_keeps_complete_sliding_window_with_intervention():
    source = _trajectory_with_one_intervention_window()

    filtered = filter_intervention_replay_trajectories([source])

    assert len(filtered) == 1
    selected = filtered[0]
    assert selected.actions.shape == (1, 1, 8)
    assert torch.equal(selected.actions[:, 0], source.actions[1:2, 0])
    assert torch.equal(selected.rewards[:, 0], source.rewards[1:2, 0])
    assert torch.equal(selected.curr_obs["z_rl"][:, 0], source.curr_obs["z_rl"][1:2, 0])
    assert selected.intervene_flags.any()


def test_filter_drops_trajectory_without_intervention():
    source = _trajectory_with_one_intervention_window()
    source.intervene_flags.zero_()

    assert filter_intervention_replay_trajectories([source]) == []


def test_sliding_builder_backfills_only_intervention_window_anchors():
    builder_kwargs = {
        "chunk_len": 2,
        "action_dim": 2,
        "stride": 1,
        "max_episode_length": 4,
        "intervention_only": True,
    }
    if "replay_kind" in inspect.signature(RLTRawReplayBuilder).parameters:
        builder_kwargs["replay_kind"] = "rlt_ac"
    builder = RLTRawReplayBuilder(
        **builder_kwargs,
    )
    for frame in range(4):
        builder._actions.append(torch.full((1, 2), float(frame)))
        builder._rewards.append(torch.zeros(1))
        builder._terminations.append(torch.zeros(1, dtype=torch.bool))
        builder._truncations.append(torch.zeros(1, dtype=torch.bool))
        builder._dones.append(torch.zeros(1, dtype=torch.bool))
        builder._human_actions.append(torch.full((1, 2), float(frame + 10)))
        builder._human_flags.append(torch.tensor([frame == 2]))
        builder._intervene_flags.append(torch.tensor([frame == 2]))
        builder._versions.append(torch.zeros((1, 1)))
        builder._record_transition.append(torch.ones((1, 1), dtype=torch.bool))
    for frame in range(5):
        obs = {"state": torch.tensor([[frame]])}
        builder._state_obs[frame] = obs
        builder._transition_obs[frame] = obs

    request = builder.build_request()

    assert len(request.windows) == 2
    assert [window.curr_anchor for window in request.windows] == [0, 2]
    assert [window.next_anchor for window in request.windows] == [1, 3]
    assert request.anchor_count == 4
    assert request.raw_anchor_positions == [0, 1, 2, 3]
