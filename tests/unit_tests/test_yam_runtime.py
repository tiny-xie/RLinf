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

"""Safety-boundary tests for the single-writer YAM control runtime."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.yam import i2rt_backend
from rlinf.envs.realworld.yam.config import DualYamJointEnvConfig
from rlinf.envs.realworld.yam.control_runtime import YamControlRuntime
from rlinf.envs.realworld.yam.types import YamArmState, YamLeaderState


class _Follower:
    def __init__(self, device, clock, events):
        self.name = device.name
        self.action = np.asarray(device.initial_action, dtype=np.float64).copy()
        self._fail_connect = bool(device.fail_connect)
        self._clock = clock
        self._events = events
        self.connected = False
        self.timestamp_s = None
        self.commands = []
        self.hold_calls = 0
        self.close_calls = 0

    def connect(self):
        self._events.append(("connect", self.name))
        if self._fail_connect:
            raise RuntimeError(f"failed to connect {self.name}")
        self.connected = True

    def read_state(self):
        assert self.connected
        timestamp_s = self._clock() if self.timestamp_s is None else self.timestamp_s
        return YamArmState(self.action[:6], self.action[6], timestamp_s)

    def command(self, target):
        assert self.connected
        target = np.asarray(target, dtype=np.float64).copy()
        self.commands.append(target)
        self.action = target

    def joint_limits(self):
        return np.column_stack([np.full(6, -np.pi), np.full(6, np.pi)])

    def hold(self):
        assert self.connected
        self.hold_calls += 1
        self._events.append(("hold", self.name))

    def assert_healthy(self, max_feedback_age_s):
        del max_feedback_age_s
        assert self.connected

    def close(self):
        self.close_calls += 1
        self.connected = False
        self._events.append(("close", self.name))


class _Leader:
    def __init__(self, device, clock, events):
        self.name = device.name
        self.action = np.asarray(device.initial_action, dtype=np.float64).copy()
        self.buttons = tuple(device.buttons)
        self._clock = clock
        self._events = events
        self.connected = False
        self.close_calls = 0

    def connect(self):
        self.connected = True
        self._events.append(("connect", self.name))

    def read_state(self):
        assert self.connected
        arm = YamArmState(self.action[:6], self.action[6], self._clock())
        return YamLeaderState(arm=arm, buttons=self.buttons)

    def command_feedback(self, follower_joints):
        del follower_joints
        assert self.connected

    def release_feedback(self):
        return None

    def assert_healthy(self, max_feedback_age_s):
        del max_feedback_age_s
        assert self.connected

    def close(self):
        self.close_calls += 1
        self.connected = False
        self._events.append(("close", self.name))


class _Factory:
    def __init__(self, clock):
        self.clock = clock
        self.events = []
        self.followers = []
        self.leaders = []

    def create_follower(self, device):
        self.events.append(("create", device.name))
        backend = _Follower(device, self.clock, self.events)
        self.followers.append(backend)
        return backend

    def create_leader(self, device):
        self.events.append(("create", device.name))
        backend = _Leader(device, self.clock, self.events)
        self.leaders.append(backend)
        return backend


class _Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _device(name: str, action, buttons=(False, False), *, fail_connect=False):
    return SimpleNamespace(
        name=name,
        initial_action=np.asarray(action, dtype=np.float64),
        buttons=buttons,
        fail_connect=fail_connect,
    )


def _hardware(left_action=None, right_action=None, *, right_follower_fails=False):
    left_action = np.zeros(7) if left_action is None else left_action
    right_action = np.zeros(7) if right_action is None else right_action
    return SimpleNamespace(
        left_follower=_device("left_follower", left_action),
        right_follower=_device(
            "right_follower", right_action, fail_connect=right_follower_fails
        ),
        left_leader=_device("left_leader", np.zeros(7)),
        right_leader=_device("right_leader", np.zeros(7)),
    )


def _runtime(config=None, hardware=None):
    clock = _Clock()
    factory = _Factory(clock)
    runtime = YamControlRuntime(
        config or DualYamJointEnvConfig(),
        hardware or _hardware(),
        factory,
        clock=clock,
        sleeper=lambda _seconds: None,
    )
    return runtime, factory


def test_runtime_construction_is_disconnected_and_state_order_is_fixed():
    left = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    right = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, 0.25])
    runtime, factory = _runtime(hardware=_hardware(left, right))

    assert factory.events == []
    assert not runtime.followers_connected
    assert not runtime.leaders_connected

    runtime.connect_followers()

    np.testing.assert_allclose(
        runtime.read_state().as_vector(), np.concatenate([left, right])
    )
    assert [backend.hold_calls for backend in factory.followers] == [1, 1]
    assert factory.events.index(("hold", "left_follower")) < factory.events.index(
        ("create", "right_follower")
    )
    assert [backend.name for backend in factory.followers] == [
        "left_follower",
        "right_follower",
    ]


def test_follower_connection_failure_closes_both_created_backends():
    runtime, factory = _runtime(hardware=_hardware(right_follower_fails=True))

    with pytest.raises(RuntimeError, match="failed to connect right_follower"):
        runtime.connect_followers()

    assert not runtime.followers_connected
    assert len(factory.followers) == 2
    assert [backend.close_calls for backend in factory.followers] == [1, 1]


@pytest.mark.parametrize(
    "invalid_action",
    [np.zeros((2, 7), dtype=np.float64), np.zeros(13, dtype=np.float64)],
    ids=["matrix", "wrong_length"],
)
def test_invalid_action_shape_is_rejected_before_any_write(invalid_action):
    runtime, factory = _runtime()
    runtime.connect_followers()

    with pytest.raises(ValueError, match="shape"):
        runtime.command(invalid_action)

    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [1, 1]


def test_command_applies_joint_limits_slew_limits_and_gripper_bounds():
    config = DualYamJointEnvConfig(
        max_joint_delta=0.2,
        joint_limit_min=[[-1.0] * 6, [-0.1, -1.0, -1.0, -1.0, -1.0, -1.0]],
        joint_limit_max=[[0.1, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0] * 6],
    )
    runtime, factory = _runtime(config=config)
    runtime.connect_followers()
    requested = np.array(
        [5.0, 5.0, -5.0, -5.0, 0.05, -0.05, 2.0]
        + [-5.0, -5.0, 5.0, 5.0, 0.05, -0.05, -1.0]
    )

    result = runtime.command(requested)

    expected_left = np.array([0.1, 0.2, -0.2, -0.2, 0.05, -0.05, 1.0])
    expected_right = np.array([-0.1, -0.2, 0.2, 0.2, 0.05, -0.05, 0.0])
    assert result.clipped
    assert result.rejection_reason is None
    np.testing.assert_allclose(
        result.accepted, np.concatenate([expected_left, expected_right])
    )
    np.testing.assert_allclose(factory.followers[0].commands, [expected_left])
    np.testing.assert_allclose(factory.followers[1].commands, [expected_right])


def test_non_finite_action_holds_measured_pose_without_forwarding_the_action():
    left = np.array([0.2, 0.1, 0.0, -0.1, -0.2, -0.3, 0.8])
    right = np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    runtime, factory = _runtime(hardware=_hardware(left, right))
    runtime.connect_followers()
    invalid = np.concatenate([left, right])
    invalid[3] = np.nan

    result = runtime.command(invalid)

    assert result.rejection_reason == "non_finite_action"
    assert not result.clipped
    np.testing.assert_allclose(result.accepted, np.concatenate([left, right]))
    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [2, 2]


def test_stale_feedback_holds_both_followers_without_writing_targets():
    runtime, factory = _runtime(config=DualYamJointEnvConfig(feedback_timeout_s=0.25))
    runtime.connect_followers()
    factory.followers[0].timestamp_s = 99.0

    with pytest.raises(RuntimeError, match="stale YAM follower feedback"):
        runtime.command(np.zeros(14))

    assert factory.followers[0].commands == []
    assert factory.followers[1].commands == []
    assert [backend.hold_calls for backend in factory.followers] == [2, 2]


def test_close_releases_each_backend_once_in_safe_reverse_order():
    runtime, factory = _runtime()
    runtime.connect_followers()
    runtime.connect_leaders()

    runtime.close()
    runtime.close()

    assert [backend.close_calls for backend in factory.followers] == [1, 1]
    assert [backend.close_calls for backend in factory.leaders] == [1, 1]
    close_order = [name for event, name in factory.events if event == "close"]
    assert close_order == [
        "right_leader",
        "left_leader",
        "right_follower",
        "left_follower",
    ]


def test_i2rt_backends_select_safe_role_specific_startup_modes(monkeypatch):
    startup_modes = []

    class _Robot:
        def get_robot_info(self):
            return {"kp": np.ones(6)}

        def update_kp_kd(self, *, kp, kd):
            del kp, kd

    def build_robot(device_config, *, zero_gravity_mode):
        del device_config
        startup_modes.append(zero_gravity_mode)
        return _Robot()

    monkeypatch.setattr(i2rt_backend, "_build_yam", build_robot)
    device = SimpleNamespace(bilateral_kp=0.0)

    i2rt_backend.I2RTYamFollower(device).connect()
    i2rt_backend.I2RTYamLeader(device).connect()

    assert startup_modes == [False, True]
