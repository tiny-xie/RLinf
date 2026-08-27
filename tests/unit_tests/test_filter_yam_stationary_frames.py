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

"""Tests for the dual-YAM stationary-frame filter."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

_TOOLKIT_DIR = Path(__file__).resolve().parents[2] / "toolkits" / "lerobot"
sys.path.insert(0, str(_TOOLKIT_DIR))
yam_filter = importlib.import_module("filter_yam_stationary_frames")


def _mask(
    actions,
    states,
    done,
    keep_every=3,
    action_joint_epsilon=1e-5,
    state_joint_epsilon=1e-4,
):
    return yam_filter.build_keep_mask(
        actions,
        states,
        done,
        action_joint_epsilon=action_joint_epsilon,
        action_gripper_epsilon=1e-5,
        state_joint_epsilon=state_joint_epsilon,
        state_gripper_epsilon=1e-4,
        stationary_keep_every=keep_every,
    )


def test_moving_follower_is_kept_when_action_is_constant():
    actions = [np.zeros(14) for _ in range(7)]
    states = [np.zeros(14) for _ in range(7)]
    states[1][0] = 0.01
    states[2][0] = 0.02
    for index in range(3, 7):
        states[index][0] = 0.02

    mask = _mask(actions, states, [[False]] * 6 + [[True]])

    assert mask == [True, True, True, False, False, True, True]


def test_gripper_only_action_is_not_filtered():
    actions = [np.zeros(14) for _ in range(4)]
    actions[2][6] = 1e-3
    states = [np.zeros(14) for _ in range(4)]

    assert _mask(actions, states, [[False]] * 3 + [[True]])[2] is True


def test_singleton_done_column_is_reduced_by_value():
    assert yam_filter._done([False]) is False
    assert yam_filter._done([True]) is True


def test_per_frame_threshold_downsamples_subthreshold_motion():
    actions = [np.zeros(14) for _ in range(5)]
    for index, action in enumerate(actions):
        action[0] = index * 4e-4
    states = [action.copy() for action in actions]

    mask = _mask(
        actions,
        states,
        [[False]] * 4 + [[True]],
        keep_every=100,
        action_joint_epsilon=5e-4,
        state_joint_epsilon=5e-4,
    )

    assert mask == [True, False, False, False, True]


def test_sustained_hold_keeps_boundaries_and_periodic_samples():
    actions = [np.zeros(14) for _ in range(10)]
    states = [np.zeros(14) for _ in range(10)]

    mask = _mask(
        actions,
        states,
        [[False]] * 9 + [[True]],
        keep_every=3,
    )

    assert mask == [True, False, False, True, False, False, True, False, False, True]
