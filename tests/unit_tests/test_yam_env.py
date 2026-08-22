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

"""Hardware-free Gym contract tests for the dual-YAM environment."""

from __future__ import annotations

import numpy as np

from rlinf.envs.realworld.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.realworld.yam.types import (
    DualYamState,
    YamArmState,
    YamCommandResult,
    split_dual_action,
)


class _Runtime:
    def __init__(self, state_vector):
        self.state = np.asarray(state_vector, dtype=np.float64).copy()
        self.connect_calls = 0
        self.read_calls = 0
        self.commands = []
        self.hold_calls = 0
        self.close_calls = 0

    def connect_followers(self):
        self.connect_calls += 1

    def read_state(self):
        self.read_calls += 1
        left, right = split_dual_action(self.state)
        return DualYamState(
            left=YamArmState(left[:6], left[6], 1.0),
            right=YamArmState(right[:6], right[6], 1.0),
        )

    def command(self, action):
        requested = np.asarray(action, dtype=np.float64).copy()
        self.commands.append(requested)
        self.state = requested
        return YamCommandResult(requested=requested, accepted=requested)

    def hold(self):
        self.hold_calls += 1
        return self.state.copy()

    def emergency_hold(self):
        self.hold_calls += 1

    def close(self):
        self.close_calls += 1


def _camera_must_not_be_created(_camera_info):
    raise AssertionError("dummy YAM env attempted to create a camera")


def test_dummy_reset_is_lazy_and_returns_the_canonical_observation():
    initial = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
        + [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.25]
    )
    runtime = _Runtime(initial)
    env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "image_height": 4,
            "image_width": 5,
            "dummy_camera_names": ["top_rgb", "left_rgb", "right_rgb"],
            "task_description": "pick the block",
        },
        runtime=runtime,
        camera_factory=_camera_must_not_be_created,
    )

    assert runtime.connect_calls == 0
    assert runtime.read_calls == 0
    assert env.action_space.shape == (14,)
    assert env.task_description == "pick the block"

    observation, info = env.reset()

    assert runtime.connect_calls == 1
    assert info == {"episode_phase": "pre"}
    np.testing.assert_allclose(observation["state"]["joint_position"], initial)
    assert list(observation["frames"]) == ["top_rgb", "left_rgb", "right_rgb"]
    assert all(
        frame.shape == (4, 5, 3) and not frame.any()
        for frame in observation["frames"].values()
    )
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(env.get_hold_action(), initial)


def test_dummy_step_preserves_14d_order_and_close_is_idempotent(monkeypatch):
    runtime = _Runtime(np.zeros(14))
    env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "max_num_steps": 1,
            "dummy_camera_names": ["top_rgb"],
        },
        runtime=runtime,
        camera_factory=_camera_must_not_be_created,
    )
    monkeypatch.setattr(env, "_pace", lambda: None)
    env.reset()
    action = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] + [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.2],
        dtype=np.float32,
    )

    observation, reward, terminated, truncated, info = env.step(action)

    np.testing.assert_allclose(runtime.commands, [action])
    np.testing.assert_allclose(observation["state"]["joint_position"], action)
    np.testing.assert_allclose(info["accepted_action"], action)
    assert reward == 0.0
    assert not terminated
    assert truncated

    env.close()
    env.close()
    assert runtime.close_calls == 1
