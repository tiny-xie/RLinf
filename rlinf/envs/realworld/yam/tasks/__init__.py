# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gym registration and factory for RLinf-native YAM tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.realworld.yam.leader_intervention import DualYamLeaderIntervention


def create_dual_yam_joint_env(
    override_cfg: dict[str, Any],
    worker_info: Any = None,
    hardware_info: Any = None,
    env_idx: int = 0,
    env_cfg: Mapping[str, Any] | None = None,
) -> gym.Env:
    """Build the base follower env and optional motorized-leader wrapper."""
    base_config = dict(override_cfg or {})
    leader_config_value = base_config.pop(
        "leader_intervention", base_config.pop("yam_leader", {})
    )
    leader_config = dict(leader_config_value or {})
    use_leaders = bool(
        base_config.pop(
            "use_yam_leader",
            leader_config.pop("enabled", False),
        )
    )
    env = DualYamJointEnv(
        override_cfg=base_config,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )

    if env_cfg is not None:
        main_image_key = env_cfg.get("main_image_key")
        if (
            main_image_key is not None
            and main_image_key not in env.observation_space["frames"]
        ):
            env.close()
            raise ValueError(
                f"YAM main_image_key {main_image_key!r} is not configured; "
                f"available cameras: {list(env.observation_space['frames'])}"
            )
    if use_leaders:
        return DualYamLeaderIntervention(env, leader_config)
    return env


register(
    id="DualYamJointEnv-v1",
    entry_point="rlinf.envs.realworld.yam.tasks:create_dual_yam_joint_env",
)


__all__ = ["create_dual_yam_joint_env"]
