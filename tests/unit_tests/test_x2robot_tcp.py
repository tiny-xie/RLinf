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
from collections import deque

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from rlinf.envs import SupportedEnvType, get_env_cls  # noqa: E402
from rlinf.envs.realworld.x2robot.realworld_env import (  # noqa: E402
    X2RobotTCPRealWorldEnv,
)
from rlinf.envs.realworld.x2robot.tcp_env import (
    X2RobotTCPConfig,
    X2RobotTCPEnv,
    _blend_chunk_transition,
)
from rlinf.envs.realworld.x2robot.upload_server import UploadServer  # noqa: E402


def test_x2robot_env_type_is_registered():
    assert SupportedEnvType.X2ROBOT_TCP.value == "x2robot_tcp"
    assert get_env_cls("x2robot_tcp") is X2RobotTCPRealWorldEnv


def test_x2robot_config_derives_sequence_geometry():
    config = X2RobotTCPConfig(
        state_history_size=3,
        state_future_size=2,
        latency_step=None,
    )

    assert config.state_seq_len == 6
    assert config.latency_step == 2
    assert config.latency_len == 6

    with pytest.raises(ValueError, match="unsupported policy_mode"):
        X2RobotTCPConfig(policy_mode="invalid")


def test_x2robot_assembles_state_and_serializes_action_chunk():
    env = object.__new__(X2RobotTCPEnv)
    env.config = X2RobotTCPConfig()
    master_queue = deque(maxlen=100)
    left = np.arange(28, dtype=np.float32).reshape(4, 7)
    right = left + 100

    state = env._assemble_state((left, right), master_queue)

    assert state.shape == (6, 32)
    np.testing.assert_array_equal(state[3, :7], left[-1])
    np.testing.assert_array_equal(state[3, 7:14], right[-1])
    expected_master = np.repeat(state[-1:, :14], state.shape[0], axis=0)
    np.testing.assert_array_equal(state[:, 14:28], expected_master)

    chunk = np.arange(20 * 28, dtype=np.float32).reshape(20, 28)
    payload = env._postprocess(chunk, master_queue)

    command = json.loads(payload)
    assert len(command["follow1_pos"]) == env.config.move_steps + 1
    assert len(command["follow2_pos"]) == env.config.move_steps + 1
    np.testing.assert_array_equal(
        command["follow1_pos"][1], chunk[env.config.latency_step, 14:21]
    )


def test_x2robot_blending_preserves_gripper_dimensions():
    action_chunk = np.array([[0.0, 5.0], [10.0, 6.0], [20.0, 7.0]], dtype=np.float64)
    history = deque([np.array([-1.0, 4.0]), np.array([0.0, 5.0])])

    blended = _blend_chunk_transition(
        action_chunk, history, blend_steps=2, skip_dims=(1,)
    )

    assert blended[1, 0] != action_chunk[1, 0]
    np.testing.assert_array_equal(blended[:, 1], action_chunk[:, 1])


def test_upload_record_keeps_executed_action_without_decoding_images():
    server = UploadServer(decode_images=False)
    header = {
        "seq": 7,
        "t": 1.25,
        "mode": 2,
        "is_takeover": True,
        "follow1_pos": list(range(7)),
        "follow2_pos": list(range(7, 14)),
        "master1_pos": list(range(14, 21)),
        "master2_pos": list(range(21, 28)),
    }

    record = server._build_record(header, [b"", b"", b""])

    assert record["is_takeover"] is True
    assert record["action_28"].shape == (28,)
    assert "frames" not in record
