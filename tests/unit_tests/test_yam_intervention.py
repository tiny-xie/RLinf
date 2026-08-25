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

"""Policy/leader ownership tests for the dual-YAM intervention wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.realworld.yam.leader_intervention import (
    DualYamLeaderIntervention,
)
from rlinf.envs.realworld.yam.types import (
    DualYamState,
    YamArmState,
    YamCommandResult,
    split_dual_action,
)


class _Runtime:
    def __init__(self) -> None:
        self.state = np.zeros(14, dtype=np.float64)
        self.leader_action = np.zeros(14, dtype=np.float64)
        self.buttons = (False, False)
        self.commands: list[np.ndarray] = []
        self.hold_calls = 0
        self.emergency_hold_calls = 0
        self.release_calls = 0
        self.engage_calls = 0
        self.close_calls = 0
        self.fail_hold = False

    def connect_followers(self) -> None:
        return None

    def connect_leaders(self) -> None:
        return None

    def read_state(self) -> DualYamState:
        left, right = split_dual_action(self.state)
        return DualYamState(
            left=YamArmState(left[:6], left[6], 1.0),
            right=YamArmState(right[:6], right[6], 1.0),
        )

    def read_leader_action(self) -> tuple[np.ndarray, tuple[bool, bool]]:
        return self.leader_action.copy(), self.buttons

    def command(self, action: np.ndarray) -> YamCommandResult:
        requested = np.asarray(action, dtype=np.float64).copy()
        self.commands.append(requested)
        self.state = requested
        return YamCommandResult(requested=requested, accepted=requested)

    def hold(self) -> np.ndarray:
        self.hold_calls += 1
        if self.fail_hold:
            raise RuntimeError("hold failed")
        return self.state.copy()

    def emergency_hold(self) -> None:
        self.emergency_hold_calls += 1

    def engage(self) -> YamCommandResult:
        self.engage_calls += 1
        self.state = self.leader_action.copy()
        return YamCommandResult(
            requested=self.leader_action.copy(),
            accepted=self.leader_action.copy(),
        )

    def apply_leader_feedback(self) -> None:
        return None

    def release_leader_feedback(self) -> None:
        self.release_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_policy_passthrough_and_leader_takeover_have_distinct_info(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "unsynced_action_source": "policy",
        },
    )
    env.reset()
    policy_action = np.linspace(0.0, 0.13, 14)

    _, _, _, _, policy_info = env.step(policy_action)

    np.testing.assert_allclose(runtime.commands[-1], policy_action)
    assert "intervene_action" not in policy_info
    assert not policy_info["intervened"]

    runtime.buttons = (True, False)
    runtime.leader_action = np.linspace(0.2, 0.33, 14)
    _, _, _, _, leader_info = env.step(policy_action)

    assert runtime.engage_calls == 1
    np.testing.assert_allclose(runtime.commands[-1], runtime.leader_action)
    np.testing.assert_allclose(
        leader_info["intervene_action"], runtime.leader_action.astype(np.float32)
    )
    assert leader_info["intervened"]

    runtime.buttons = (False, False)
    env.step(policy_action)
    runtime.buttons = (True, False)
    env.step(policy_action)

    assert runtime.release_calls == 2  # reset plus sync-off
    assert not env._sync_enabled


def test_sync_off_failure_still_releases_feedback_and_clears_ownership(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "unsynced_action_source": "policy",
        },
    )
    env.reset()

    runtime.buttons = (True, False)
    env.step(np.zeros(14))
    assert env._sync_enabled

    runtime.buttons = (False, False)
    env.step(np.zeros(14))
    runtime.buttons = (True, False)
    runtime.fail_hold = True

    with pytest.raises(RuntimeError, match="hold failed"):
        env.step(np.zeros(14))

    assert not env._sync_enabled
    assert runtime.emergency_hold_calls == 1
    assert runtime.release_calls == 3  # reset, sync-off finally, error fallback


def test_base_done_immediately_releases_leader_feedback(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={
            "is_dummy": True,
            "dummy_camera_names": ["top_rgb"],
            "max_num_steps": 1,
        },
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "unsynced_action_source": "policy",
        },
    )
    env.reset()
    runtime.buttons = (True, False)

    _, _, terminated, truncated, info = env.step(np.zeros(14))

    assert not terminated
    assert truncated
    assert not env._sync_enabled
    assert not info["yam_sync_enabled"]
    assert runtime.release_calls == 2  # reset plus done handoff


def test_manual_record_boundary_preserves_sync_across_reset(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "preserve_sync_between_episodes": True,
            "unsynced_action_source": "hold",
        },
    )
    env.reset()

    runtime.buttons = (True, False)
    env.step(np.zeros(14))
    runtime.buttons = (False, False)
    env.step(np.zeros(14))
    runtime.buttons = (False, True)

    _, _, terminated, truncated, info = env.step(np.zeros(14))

    assert terminated
    assert not truncated
    assert info["manual_done"]
    assert info["yam_sync_enabled"]
    assert env._sync_enabled
    assert runtime.release_calls == 1

    env.reset()

    assert env._sync_enabled
    assert runtime.engage_calls == 1
    assert runtime.release_calls == 1


def test_early_reset_still_releases_sync_in_continuous_mode(monkeypatch):
    runtime = _Runtime()
    base_env = DualYamJointEnv(
        override_cfg={"is_dummy": True, "dummy_camera_names": ["top_rgb"]},
        runtime=runtime,
    )
    monkeypatch.setattr(base_env, "_pace", lambda: None)
    env = DualYamLeaderIntervention(
        base_env,
        {
            "wait_for_record_button": False,
            "button_debounce_s": 0.0,
            "preserve_sync_between_episodes": True,
            "unsynced_action_source": "hold",
        },
    )
    env.reset()
    runtime.buttons = (True, False)
    env.step(np.zeros(14))

    env.reset()

    assert not env._sync_enabled
    # One hold initializes the base environment; the second safely disables
    # synchronization when reset occurs before an episode boundary.
    assert runtime.hold_calls == 2
    assert runtime.release_calls == 3
